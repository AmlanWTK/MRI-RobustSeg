"""
╔══════════════════════════════════════════════════════════════════════╗
║   HD95 FULL EVALUATION SCRIPT — NETCRYPT 2026 PAPER                ║
║   Computes: DSC, HD95, IoU, Sensitivity, Specificity               ║
║   Regions : Whole Tumor (WT), Tumor Core (TC), Enhancing (ET)      ║
║   Output  : Per-patient CSV + summary table ready for paper         ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE ON KAGGLE:
  1. Add your trained checkpoint as a dataset input
  2. Add BraTS2020 dataset as input
  3. Run all cells — results saved to /kaggle/working/

HOW TO USE LOCALLY:
  1. Set CHECKPOINT_PATH and BRATS_ROOT below
  2. pip install medpy nibabel scipy tqdm pandas
  3. python evaluate_hd95.py
"""

import os, random, torch, torch.nn as nn
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42  # MUST match training seed for same train/val split

# --- Paths (update for your environment) ---
# Kaggle:
CHECKPOINT_PATH = "/kaggle/working/phd_best_checkpoint_run3.pth"
BRATS_ROOT      = None  # Auto-detected below

# Local (override if running locally):
# CHECKPOINT_PATH = r"D:\Downloads\MedDA-Old2Modern-main\MedDA-Old2Modern-main\phd_best_checkpoint.pth"
# BRATS_ROOT      = r"/path/to/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Output
OUTPUT_CSV_PER_PATIENT = "/kaggle/working/eval_per_patient.csv"
OUTPUT_CSV_SUMMARY     = "/kaggle/working/eval_summary.csv"
OUTPUT_TXT_PAPER_TABLE = "/kaggle/working/paper_results_table.txt"

BATCH_SIZE  = 8
NUM_WORKERS = 2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. MODEL DEFINITION (must match training exactly)
# ============================================================
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_l  = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi  = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_l(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionUNetPlusPlus(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.c00 = ConvBlock(in_ch, f[0]); self.c10 = ConvBlock(f[0], f[1])
        self.c20 = ConvBlock(f[1], f[2]); self.c30 = ConvBlock(f[2], f[3])
        self.c40 = ConvBlock(f[3], f[4]); self.p   = nn.MaxPool2d(2)
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
        g4 = self.up4(x40);  u4 = self.d4(torch.cat([g4, self.att4(g4, x30)], 1))
        g3 = self.up3(u4);   u3 = self.d3(torch.cat([g3, self.att3(g3, x20)], 1))
        g2 = self.up2(u3);   u2 = self.d2(torch.cat([g2, self.att2(g2, x10)], 1))
        g1 = self.up1(u2);   u1 = self.d1(torch.cat([g1, self.att1(g1, x00)], 1))
        return self.final(u1)


# ============================================================
# 2. NORMALIZATION
# ============================================================
def z_score_normalize(img):
    mask = img > 0
    if not np.any(mask):
        return img
    mean, std = img[mask].mean(), img[mask].std()
    out = np.zeros_like(img)
    out[mask] = (img[mask] - mean) / (std + 1e-8)
    return out


# ============================================================
# 3. METRICS
# ============================================================
def compute_dice(pred_bin, gt_bin):
    """Dice Similarity Coefficient."""
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    if union == 0:
        return 1.0  # Both empty → perfect score
    return (2.0 * inter) / union


def compute_iou(pred_bin, gt_bin):
    """Intersection over Union."""
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return inter / union


def compute_sensitivity(pred_bin, gt_bin):
    """Sensitivity / Recall = TP / (TP + FN)."""
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    if (tp + fn) == 0:
        return 1.0
    return tp / (tp + fn)


def compute_specificity(pred_bin, gt_bin):
    """Specificity = TN / (TN + FP)."""
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    if (tn + fp) == 0:
        return 1.0
    return tn / (tn + fp)


def compute_hd95(pred_bin, gt_bin):
    """
    Hausdorff Distance at 95th Percentile (HD95).
    Works on 3D volumes. Pure NumPy/SciPy — no medpy needed.
    Returns distance in voxels (multiply by voxel spacing for mm).
    Returns 0.0 if both are empty, 373.0 (diagonal) if one is empty.
    """
    pred_bool = pred_bin.astype(bool)
    gt_bool   = gt_bin.astype(bool)

    # Handle edge cases
    if not pred_bool.any() and not gt_bool.any():
        return 0.0
    if not pred_bool.any() or not gt_bool.any():
        # Penalize harshly — max possible distance for 240x240x155 volume
        return np.sqrt(240**2 + 240**2 + 155**2)

    # Surface of prediction: voxels in pred that border background
    pred_surface = pred_bool & ~_erode(pred_bool)
    gt_surface   = gt_bool   & ~_erode(gt_bool)

    # Distance transform from gt surface to everywhere
    dt_gt   = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    # Directed distances
    dist_pred_to_gt = dt_gt[pred_surface]    # dist from each pred surface pt to nearest gt surface
    dist_gt_to_pred = dt_pred[gt_surface]    # dist from each gt surface pt to nearest pred surface

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def _erode(binary_vol):
    """Fast binary erosion via scipy."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(binary_vol)


# ============================================================
# 4. PATIENT-LEVEL 3D EVALUATION
# ============================================================
def evaluate_patient_3d(patient_path, model):
    """
    Loads a full 3D volume, runs 2D slice-by-slice inference,
    reconstructs 3D volume, then computes all metrics.
    Returns a dict of metrics for this patient.
    """
    pid = os.path.basename(patient_path)

    # Load segmentation to find tumor slices
    seg_path = f"{patient_path}/{pid}_seg.nii"
    if not os.path.exists(seg_path):
        seg_path += ".gz"
    seg_3d = nib.load(seg_path).get_fdata()   # H x W x D
    H, W, D = seg_3d.shape

    # Ground truth 3D masks (binary per region)
    gt_wt = (seg_3d > 0).astype(np.uint8)
    gt_tc = ((seg_3d == 1) | (seg_3d == 4)).astype(np.uint8)
    gt_et = (seg_3d == 4).astype(np.uint8)

    # Load all 4 MRI modalities
    vols = {}
    for mod in ['flair', 't1', 't1ce', 't2']:
        path = f"{patient_path}/{pid}_{mod}.nii"
        if not os.path.exists(path):
            path += ".gz"
        vols[mod] = nib.load(path).get_fdata()

    # Inference slice-by-slice → reconstruct 3D pred
    pred_wt_3d = np.zeros((H, W, D), dtype=np.uint8)
    pred_tc_3d = np.zeros((H, W, D), dtype=np.uint8)
    pred_et_3d = np.zeros((H, W, D), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for s in range(D):
            channels = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                channels.append(z_score_normalize(vols[mod][:, :, s]))
            img_t = torch.from_numpy(
                np.stack(channels, 0).astype(np.float32)
            ).unsqueeze(0).to(DEVICE)  # 1 x 4 x H x W

            out = model(img_t)  # 1 x 3 x H x W
            probs = torch.sigmoid(out).squeeze(0).cpu().numpy()  # 3 x H x W

            pred_wt_3d[:, :, s] = (probs[0] > 0.5).astype(np.uint8)
            pred_tc_3d[:, :, s] = (probs[1] > 0.5).astype(np.uint8)
            pred_et_3d[:, :, s] = (probs[2] > 0.5).astype(np.uint8)

    # Compute all metrics per region
    results = {"patient_id": pid}
    for region, pred, gt in [
        ("WT", pred_wt_3d, gt_wt),
        ("TC", pred_tc_3d, gt_tc),
        ("ET", pred_et_3d, gt_et),
    ]:
        pred_b = pred.astype(bool)
        gt_b   = gt.astype(bool)
        results[f"dice_{region}"]        = compute_dice(pred_b, gt_b)
        results[f"iou_{region}"]         = compute_iou(pred_b, gt_b)
        results[f"sensitivity_{region}"] = compute_sensitivity(pred_b, gt_b)
        results[f"specificity_{region}"] = compute_specificity(pred_b, gt_b)
        results[f"hd95_{region}"]        = compute_hd95(pred_b, gt_b)

    return results


# ============================================================
# 5. DATA DISCOVERY
# ============================================================
def find_brats_root():
    search_paths = [
        "/kaggle/input",
        BRATS_ROOT,
    ]
    for base in search_paths:
        if base is None:
            continue
        for root, dirs, _ in os.walk(base):
            if any("BraTS20_Training_001" in d for d in dirs):
                print(f"  ✅ BraTS root: {root}")
                return root
    return None


# ============================================================
# 6. MAIN EVALUATION
# ============================================================
def run_evaluation():
    print(f"\n{'='*65}")
    print(f"  🔬 HD95 FULL EVALUATION — NETCRYPT 2026")
    print(f"  Device: {DEVICE}")
    print(f"{'='*65}\n")

    # --- Load model ---
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("   Set CHECKPOINT_PATH at the top of the script.")
        return

    print(f"🔄 Loading model from: {CHECKPOINT_PATH}")
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = AttentionUNetPlusPlus().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    trained_epoch = ckpt.get('epoch', '?')
    best_dice     = ckpt.get('best_dice', '?')
    print(f"  ✅ Loaded | Trained Epoch: {trained_epoch} | Saved Best Dice: {best_dice:.4f}\n")

    # --- Find data ---
    root = find_brats_root()
    if root is None:
        print("❌ BraTS data not found. Set BRATS_ROOT at the top of the script.")
        return

    all_p = sorted([
        os.path.join(root, d) for d in os.listdir(root)
        if "BraTS20_Training_" in d
    ])

    # Reproduce EXACT same val split as training
    random.seed(SEED)
    random.shuffle(all_p)
    split = int(len(all_p) * 0.8)
    val_p = all_p[split:]
    print(f"  Evaluating on {len(val_p)} validation patients...\n")

    # --- Run patient-level evaluation ---
    all_results = []
    failed = []

    for p_path in tqdm(val_p, desc="Evaluating patients"):
        try:
            metrics = evaluate_patient_3d(p_path, model)
            all_results.append(metrics)
        except Exception as e:
            pid = os.path.basename(p_path)
            print(f"\n  ⚠️  Skipped {pid}: {e}")
            failed.append(pid)

    if not all_results:
        print("❌ No results computed. Check your data paths.")
        return

    # --- Save per-patient CSV ---
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV_PER_PATIENT, index=False)
    print(f"\n✅ Per-patient results saved → {OUTPUT_CSV_PER_PATIENT}")

    # --- Compute summary statistics ---
    metric_cols = [c for c in df.columns if c != "patient_id"]
    summary = df[metric_cols].agg(['mean', 'std', 'median', 'min', 'max'])
    summary.to_csv(OUTPUT_CSV_SUMMARY)
    print(f"✅ Summary statistics saved → {OUTPUT_CSV_SUMMARY}")

    # --- Print paper-ready table ---
    means = df[metric_cols].mean()
    stds  = df[metric_cols].std()

    paper_table = []
    paper_table.append("=" * 70)
    paper_table.append("  RESULTS TABLE — READY FOR NETCRYPT 2026 PAPER")
    paper_table.append(f"  Model: Attention UNet++ | Epoch: {trained_epoch} | N={len(all_results)} patients")
    paper_table.append("=" * 70)
    paper_table.append(f"\n  {'Metric':<28} {'WT':>12} {'TC':>12} {'ET':>12}")
    paper_table.append(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")

    metrics_display = [
        ("DSC (Dice) ↑",      "dice"),
        ("IoU ↑",             "iou"),
        ("Sensitivity ↑",     "sensitivity"),
        ("Specificity ↑",     "specificity"),
        ("HD95 (voxels) ↓",   "hd95"),
    ]

    for label, key in metrics_display:
        row = f"  {label:<28}"
        for region in ["WT", "TC", "ET"]:
            col  = f"{key}_{region}"
            mean = means[col]
            std  = stds[col]
            if key == "hd95":
                row += f" {mean:>7.2f}±{std:<4.2f}"
            else:
                row += f" {mean:>7.4f}±{std:<4.4f}"
        paper_table.append(row)

    paper_table.append(f"\n  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
    paper_table.append(f"  Failed patients: {len(failed)}")
    paper_table.append("=" * 70)
    paper_table.append("\n  NOTE: HD95 in voxels. For mm, multiply by voxel spacing (BraTS=1mm isotropic).")
    paper_table.append("        For BraTS 2020: 1 voxel = 1 mm, so HD95 voxels = HD95 mm directly.")
    paper_table.append("=" * 70)

    table_str = "\n".join(paper_table)
    print("\n" + table_str)

    with open(OUTPUT_TXT_PAPER_TABLE, "w") as f:
        f.write(table_str)
    print(f"\n✅ Paper table saved → {OUTPUT_TXT_PAPER_TABLE}")

    # --- Quick Dice check vs 2D average (sanity) ---
    print(f"\n📊 Quick Sanity Check (3D patient-level vs 2D slice-level from training):")
    print(f"   3D Val WT Dice: {means['dice_WT']:.4f} ± {stds['dice_WT']:.4f}")
    print(f"   3D Val TC Dice: {means['dice_TC']:.4f} ± {stds['dice_TC']:.4f}")
    print(f"   3D Val ET Dice: {means['dice_ET']:.4f} ± {stds['dice_ET']:.4f}")
    print(f"   HD95 WT (mm)  : {means['hd95_WT']:.2f} ± {stds['hd95_WT']:.2f}")
    print(f"   HD95 TC (mm)  : {means['hd95_TC']:.2f} ± {stds['hd95_TC']:.2f}")
    print(f"   HD95 ET (mm)  : {means['hd95_ET']:.2f} ± {stds['hd95_ET']:.2f}")
    print(f"\n🎉 Evaluation complete! All files saved to /kaggle/working/")


run_evaluation()

