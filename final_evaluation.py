"""
IEEE Publication Codebase: Final 3D Volumetric Evaluation Script
Target Conference: NETCRYPT 2026
Description: This script performs 3D patient-level evaluation for the two
remaining ablation configurations (Att-UNet++ without augmentation, and the
Proposed Framework) and combines all results into the final publication table.

INSTRUCTIONS:
1. Upload this script to a new Kaggle notebook.
2. Add the following datasets as inputs:
   - Your BraTS 2020 dataset
   - Your checkpoint dataset containing:
       * ablation_AttUNetPP_NoAug_13.pth   (Att-UNet++ No-Aug, Epoch 13)
       * phd_best_checkpoint.pth           (Proposed Framework, Epoch 22)
       * ablation_BaselineUNet_patients.csv
       * ablation_UNetPP_patients.csv
3. Run all. The script will produce the final paper table automatically.
"""

import os, torch, random
import torch.nn as nn
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =============================================================================
# SECTION 1: NORMALIZATION
# =============================================================================
def z_score_normalize(img):
    """Apply brain-masked Z-score normalization per MRI modality."""
    mask = img > 0
    if not np.any(mask): return img
    m, s = img[mask].mean(), img[mask].std()
    out = np.zeros_like(img)
    out[mask] = (img[mask] - m) / (s + 1e-8)
    return out

# =============================================================================
# SECTION 2: MODEL ARCHITECTURES
# =============================================================================
class ConvBlock(nn.Module):
    """Dual-convolutional block with Batch Normalization and Dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x): return self.conv(x)

class AttentionGate(nn.Module):
    """Soft-attention gate for selective feature suppression."""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.Wl = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(True)
    def forward(self, g, x): return x * self.psi(self.relu(self.Wg(g) + self.Wl(x)))

class AttentionUNetPlusPlus(nn.Module):
    """
    Proposed Architecture: Attention U-Net++ for Multimodal Brain Tumor Segmentation.
    Used for both Ablation Variant 3 (without augmentation) and the Proposed Framework.
    Reference: Zhou et al. (2018) + Oktay et al. (2018) - combined implementation.
    Input:  4-channel MRI (FLAIR, T1, T1ce, T2), shape (B, 4, H, W)
    Output: 3-channel binary mask (WT, TC, ET), shape (B, 3, H, W)
    """
    def __init__(self, in_ch=4, out_ch=3):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.c00 = ConvBlock(in_ch, f[0]); self.c10 = ConvBlock(f[0], f[1])
        self.c20 = ConvBlock(f[1], f[2]); self.c30 = ConvBlock(f[2], f[3])
        self.c40 = ConvBlock(f[3], f[4]); self.p = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2)
        self.att4 = AttentionGate(f[3], f[3], f[2]); self.d4 = ConvBlock(f[3]*2, f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.att3 = AttentionGate(f[2], f[2], f[1]); self.d3 = ConvBlock(f[2]*2, f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.att2 = AttentionGate(f[1], f[1], f[0]); self.d2 = ConvBlock(f[1]*2, f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.att1 = AttentionGate(f[0], f[0], f[0]//2); self.d1 = ConvBlock(f[0]*2, f[0])
        self.final = nn.Conv2d(f[0], out_ch, 1)
    def forward(self, x):
        x00 = self.c00(x); x10 = self.c10(self.p(x00))
        x20 = self.c20(self.p(x10)); x30 = self.c30(self.p(x20))
        x40 = self.c40(self.p(x30))
        g4 = self.up4(x40); u4 = self.d4(torch.cat([g4, self.att4(g4, x30)], 1))
        g3 = self.up3(u4);  u3 = self.d3(torch.cat([g3, self.att3(g3, x20)], 1))
        g2 = self.up2(u3);  u2 = self.d2(torch.cat([g2, self.att2(g2, x10)], 1))
        g1 = self.up1(u2);  u1 = self.d1(torch.cat([g1, self.att1(g1, x00)], 1))
        return self.final(u1)

# =============================================================================
# SECTION 3: EVALUATION METRICS
# =============================================================================
def compute_dice_3d(pred, gt):
    """Compute volumetric Dice Similarity Coefficient (DSC)."""
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    return float(2.0 * inter / union) if union > 0 else 1.0

def compute_hd95(pred, gt):
    """Compute 95th Percentile Hausdorff Distance (HD95) in mm."""
    pb, gb = pred.astype(bool), gt.astype(bool)
    if not pb.any() and not gb.any(): return 0.0
    if not pb.any() or not gb.any(): return 373.13  # Maximum BraTS penalty
    ps = pb & ~binary_erosion(pb)
    gs = gb & ~binary_erosion(gb)
    d1 = distance_transform_edt(~gs)[ps]
    d2 = distance_transform_edt(~ps)[gs]
    return float(np.percentile(np.concatenate([d1, d2]), 95))

# =============================================================================
# SECTION 4: 3D PATIENT-LEVEL EVALUATION
# =============================================================================
def evaluate_patient_3d(patient_path, model):
    """
    Perform slice-by-slice inference on a full 3D volume and compute
    volumetric metrics (DSC and HD95) for all three tumor sub-regions.
    """
    pid = os.path.basename(patient_path)
    seg_path = f"{patient_path}/{pid}_seg.nii"
    if not os.path.exists(seg_path): seg_path += ".gz"

    seg3d = nib.load(seg_path).get_fdata()
    H, W, D = seg3d.shape

    # Ground truth sub-regions (BraTS 2020 label convention)
    gt_wt = (seg3d > 0).astype(np.uint8)
    gt_tc = ((seg3d == 1) | (seg3d == 4)).astype(np.uint8)
    gt_et = (seg3d == 4).astype(np.uint8)

    # Load MRI modalities
    vols = {}
    for mod in ['flair', 't1', 't1ce', 't2']:
        path = f"{patient_path}/{pid}_{mod}.nii"
        if not os.path.exists(path): path += ".gz"
        vols[mod] = nib.load(path).get_fdata()

    # Slice-wise inference → reconstruct 3D prediction volume
    pred_wt = np.zeros((H, W, D), dtype=np.uint8)
    pred_tc = np.zeros((H, W, D), dtype=np.uint8)
    pred_et = np.zeros((H, W, D), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for s in range(D):
            chs = [z_score_normalize(vols[m][:, :, s]) for m in ['flair', 't1', 't1ce', 't2']]
            inp = torch.from_numpy(np.stack(chs, 0).astype(np.float32)).unsqueeze(0).to(DEVICE)
            out = torch.sigmoid(model(inp)).squeeze(0).cpu().numpy()
            pred_wt[:, :, s] = (out[0] > 0.5)
            pred_tc[:, :, s] = (out[1] > 0.5)
            pred_et[:, :, s] = (out[2] > 0.5)

    # Compute metrics for all three sub-regions
    results = {"patient_id": pid}
    for region_name, pred, gt in [("WT", pred_wt, gt_wt), ("TC", pred_tc, gt_tc), ("ET", pred_et, gt_et)]:
        results[f"dice_{region_name}"] = compute_dice_3d(pred.astype(bool), gt.astype(bool))
        results[f"hd95_{region_name}"] = compute_hd95(pred, gt)
    return results

def evaluate_model(name, model, val_patients, output_csv):
    """Run full 3D evaluation over all validation patients and save results."""
    print(f"\n{'='*60}")
    print(f"  3D Volumetric Evaluation: {name}")
    print(f"  Subjects: {len(val_patients)}")
    print(f"{'='*60}")
    records = []
    for pp in tqdm(val_patients, desc=f"Evaluating {name}"):
        try:
            records.append(evaluate_patient_3d(pp, model))
        except Exception as e:
            print(f"  [WARNING] Failed on {os.path.basename(pp)}: {e}")
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    means = df[[c for c in df.columns if c != "patient_id"]].mean()
    print(f"\n  Results → Dice: WT={means['dice_WT']:.4f}, TC={means['dice_TC']:.4f}, ET={means['dice_ET']:.4f}")
    print(f"           HD95: WT={means['hd95_WT']:.2f}, TC={means['hd95_TC']:.2f}, ET={means['hd95_ET']:.2f}")
    return means

# =============================================================================
# SECTION 5: DATA DISCOVERY
# =============================================================================
def find_brats_root():
    """Auto-discover BraTS 2020 training data directory on Kaggle."""
    for root, dirs, _ in os.walk("/kaggle/input"):
        if any("BraTS20_Training_001" in d for d in dirs):
            return root
    return None

def find_file(filename, search_dirs):
    """Search for a file across multiple possible input directories."""
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

# =============================================================================
# SECTION 6: MAIN EVALUATION PIPELINE
# =============================================================================
def run_final_evaluation():
    # --- Data Setup ---
    root = find_brats_root()
    if not root:
        print("[ERROR] BraTS 2020 dataset not found in /kaggle/input. Please add it as an input dataset.")
        return

    all_patients = sorted([
        os.path.join(root, d) for d in os.listdir(root) if "BraTS20_Training_" in d
    ])
    random.seed(SEED)
    random.shuffle(all_patients)
    val_patients = all_patients[int(len(all_patients) * 0.8):]  # Same 20% val split as training
    print(f"\n✅ Found {len(val_patients)} validation subjects for 3D evaluation.")

    # --- Checkpoint Discovery ---
    # --- Checkpoint & CSV Discovery (Exact Kaggle Paths) ---
    all_results = {}
    
    PATH_BASELINE_CSV = "/kaggle/input/datasets/amlan21s/baseline/ablation_BaselineUNet_patients.csv"
    PATH_UNETPP_CSV   = "/kaggle/input/datasets/amlan21s/ablation-study/ablation_UNetPP_patients.csv"
    PATH_NOAUG_CKPT   = "/kaggle/input/datasets/amlan21s/abla-13/ablation_AttUNetPP_NoAug_13.pth"
    PATH_FULL_CKPT    = "/kaggle/input/datasets/amlan21s/latest/phd_best_checkpoint.pth"

    # --- Step 1: Load pre-computed results for Baseline UNet ---
    print("\n📂 Loading pre-computed results: Standard U-Net (Baseline)")
    baseline_csv = PATH_BASELINE_CSV
    if os.path.exists(baseline_csv):
        df_b = pd.read_csv(baseline_csv)
        all_results["Standard U-Net"] = df_b[[c for c in df_b.columns if c != "patient_id"]].mean()
        print(f"  ✅ Loaded from {baseline_csv}")
    else:
        print(f"  [WARNING] {baseline_csv} not found. Skipping.")

    # --- Step 2: Load pre-computed results for U-Net++ ---
    print("\n📂 Loading pre-computed results: U-Net++ (Nested Architecture)")
    unetpp_csv = PATH_UNETPP_CSV
    if os.path.exists(unetpp_csv):
        df_u = pd.read_csv(unetpp_csv)
        all_results["U-Net++ (Nested)"] = df_u[[c for c in df_u.columns if c != "patient_id"]].mean()
        print(f"  ✅ Loaded from {unetpp_csv}")
    else:
        print(f"  [WARNING] {unetpp_csv} not found. Skipping.")

    # --- Step 3: Evaluate Att-UNet++ WITHOUT Hybrid Degradation ---
    ckpt_noaug = PATH_NOAUG_CKPT
    if os.path.exists(ckpt_noaug):
        print(f"\n🔄 Loading Att-UNet++ (No Augmentation) from: {ckpt_noaug}")
        model_noaug = AttentionUNetPlusPlus().to(DEVICE)
        ckpt = torch.load(ckpt_noaug, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
            
        # Dynamically fix key mismatches between different training scripts
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('.W_g.', '.Wg.')
            k = k.replace('.W_l.', '.Wl.')
            new_state_dict[k] = v
            
        model_noaug.load_state_dict(new_state_dict)
        all_results["Att-UNet++ (w/o Augmentation)"] = evaluate_model(
            "Att-UNet++ (No Aug)", model_noaug, val_patients,
            "/kaggle/working/eval_AttUNetPP_NoAug_patients.csv"
        )
        del model_noaug; torch.cuda.empty_cache()
    else:
        print(f"\n[WARNING] {ckpt_noaug} not found. Skipping.")

    # --- Step 4: Evaluate Proposed Framework (Full Model) ---
    ckpt_full = PATH_FULL_CKPT
    if os.path.exists(ckpt_full):
        print(f"\n🔄 Loading Proposed Framework from: {ckpt_full}")
        model_full = AttentionUNetPlusPlus().to(DEVICE)
        ckpt = torch.load(ckpt_full, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
            
        # Dynamically fix key mismatches between different training scripts
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('.W_g.', '.Wg.')
            k = k.replace('.W_l.', '.Wl.')
            new_state_dict[k] = v
            
        model_full.load_state_dict(new_state_dict)
        all_results["Proposed Framework (Att-UNet++ + HDA)"] = evaluate_model(
            "Proposed Framework", model_full, val_patients,
            "/kaggle/working/eval_ProposedFramework_patients.csv"
        )
        del model_full; torch.cuda.empty_cache()
    else:
        print(f"\n[WARNING] {ckpt_full} not found. Skipping.")

    # ==========================================================================
    # TABLE I: QUANTITATIVE ABLATION RESULTS (for NETCRYPT Paper)
    # ==========================================================================
    print(f"\n{'='*90}")
    print(f"  TABLE I: ABLATION STUDY — QUANTITATIVE RESULTS ON BraTS 2020 DATASET")
    print(f"  Conference: NETCRYPT | Metric: Dice Similarity Coefficient (DSC) & HD95 (mm)")
    print(f"{'='*90}")
    print(f"  {'Configuration':<40} {'DSC-WT':>8} {'DSC-TC':>8} {'DSC-ET':>8} {'HD95-WT':>9} {'HD95-TC':>9} {'HD95-ET':>9}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")

    rows = []
    for name, m in all_results.items():
        print(f"  {name:<40} {m['dice_WT']:>8.4f} {m['dice_TC']:>8.4f} {m['dice_ET']:>8.4f} "
              f"{m['hd95_WT']:>9.2f} {m['hd95_TC']:>9.2f} {m['hd95_ET']:>9.2f}")
        row = {"Configuration": name}
        for k in m.index: row[k] = round(m[k], 4)
        rows.append(row)

    print(f"{'='*90}")
    print("\n  Note: HDA = Hybrid Degradation Augmentation. All metrics are 3D patient-level.")
    print(  "        DSC: higher is better. HD95: lower is better.")

    # Save final table
    pd.DataFrame(rows).to_csv("/kaggle/working/TABLE_I_ablation_results.csv", index=False)
    print("\n✅ Final results table saved → /kaggle/working/TABLE_I_ablation_results.csv")

run_final_evaluation()
