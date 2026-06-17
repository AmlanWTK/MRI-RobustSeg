"""
IEEE Publication Codebase: High-Speed Segmentation Training Pipeline (V2)
Target Conference: NETCRYPT 2026

Description:
    An optimized variant of the primary training pipeline that pre-slices
    the BraTS 2020 dataset to disk (NumPy format) prior to training,
    enabling significantly faster data loading (approximately 10× speedup)
    compared to on-the-fly NIfTI loading.

    Recommended for resource-constrained training environments (e.g.,
    Kaggle with limited GPU session time).

Usage (Kaggle):
    Run this script to:
      1. Pre-slice BraTS volumes into .npy files in /kaggle/working/fast_slices/.
      2. Train the Attention U-Net++ for 50 epochs using the pre-sliced data.
      3. Output: best_checkpoint_v2.pth and training_metrics_v2.csv
"""

import os, glob, torch, random, torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import pandas as pd


# =============================================================================
# SECTION 1: REPRODUCIBILITY
# =============================================================================
def set_seed(seed=42):
    """Ensure fully deterministic training for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# =============================================================================
# SECTION 2: DATA PRE-SLICING ENGINE
# =============================================================================
def preprocess_volumes_to_slices(root_path, output_dir):
    """
    Pre-slice BraTS 2020 NIfTI volumes into 2D NumPy arrays for fast loading.

    Iterates over all patient directories, filters tumor-positive axial slices
    (mask sum > 50 voxels), applies Z-score normalization, and saves
    image/mask pairs as .npy files.

    Args:
        root_path  (str): Path to the BraTS 2020 training root directory.
        output_dir (str): Output directory for pre-sliced .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_patients = sorted([
        os.path.join(root_path, d) for d in os.listdir(root_path)
        if "BraTS20_Training_" in d
    ])
    print(f"  Pre-slicing {len(all_patients)} patient volumes...")
    for p in tqdm(all_patients):
        pid = os.path.basename(p)
        seg_file = f"{p}/{pid}_seg.nii"
        if not os.path.exists(seg_file):
            continue
        seg_data = nib.load(seg_file).get_fdata()
        tumor_indices = np.where(np.sum(seg_data, axis=(0, 1)) > 50)[0]
        for s in tumor_indices:
            channels = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                img = nib.load(f"{p}/{pid}_{mod}.nii").get_fdata()[:, :, s]
                mask = img > 0
                if np.any(mask):
                    img[mask] = (img[mask] - img[mask].mean()) / (img[mask].std() + 1e-8)
                channels.append(img)
            seg_slice = seg_data[:, :, s]
            np.save(f"{output_dir}/{pid}_s{s}_img.npy",
                    np.stack(channels, 0).astype(np.float32))
            np.save(f"{output_dir}/{pid}_s{s}_msk.npy",
                    np.stack([(seg_slice > 0),
                               (seg_slice == 1) | (seg_slice == 4),
                               (seg_slice == 4)], 0).astype(np.uint8))


# =============================================================================
# SECTION 3: MODEL ARCHITECTURE — ATTENTION U-NET++
# =============================================================================
class AttentionGate(nn.Module):
    """Soft attention gate for skip connection feature recalibration."""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_l = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_l(x)))


class ConvBlock(nn.Module):
    """Dual-convolutional block with Batch Normalization and Dropout."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionUNetPlusPlus(nn.Module):
    """
    Proposed Architecture: Attention U-Net++ for Brain Tumor Segmentation.
    See train_segmentation.py for full architecture documentation.
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
        x00 = self.c00(x); x10 = self.c10(self.p(x00)); x20 = self.c20(self.p(x10))
        x30 = self.c30(self.p(x20)); x40 = self.c40(self.p(x30))
        g4 = self.up4(x40); u4 = self.d4(torch.cat([g4, self.att4(g4, x30)], 1))
        g3 = self.up3(u4);  u3 = self.d3(torch.cat([g3, self.att3(g3, x20)], 1))
        g2 = self.up2(u3);  u2 = self.d2(torch.cat([g2, self.att2(g2, x10)], 1))
        g1 = self.up1(u2);  u1 = self.d1(torch.cat([g1, self.att1(g1, x00)], 1))
        return self.final(u1)


# =============================================================================
# SECTION 4: PRE-SLICED DATASET
# =============================================================================
class PreSlicedDataset(Dataset):
    """
    Efficient PyTorch Dataset for pre-sliced .npy image/mask pairs.

    Args:
        files (list): Sorted list of absolute paths to *_img.npy files.
    """
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        msk_path = img_path.replace('_img.npy', '_msk.npy')
        return (torch.from_numpy(np.load(img_path)).float(),
                torch.from_numpy(np.load(msk_path)).float())


# =============================================================================
# SECTION 5: LOSS FUNCTION
# =============================================================================
def combined_loss(output, target):
    """Combined Dice + Focal loss. See train_segmentation.py for full documentation."""
    probs = torch.sigmoid(output)
    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_l = 1 - (2. * inter + 1e-5) / (union + 1e-5)
    bce = nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
    focal_l = (1 - torch.exp(-bce))**2 * bce
    return dice_l.mean() + focal_l.mean()


def dice_coeff_regional(output, target):
    """Per-region Dice DSC. Returns dict with WT, TC, ET scores."""
    p = (torch.sigmoid(output) > 0.5).float()
    res = {}
    for i, name in enumerate(["WT", "TC", "ET"]):
        inter = (p[:, i] * target[:, i]).sum(dim=(1, 2))
        union = p[:, i].sum(dim=(1, 2)) + target[:, i].sum(dim=(1, 2))
        res[name] = ((2. * inter + 1e-5) / (union + 1e-5)).mean().item()
    return res


# =============================================================================
# SECTION 6: MAIN TRAINING PIPELINE (HIGH-SPEED)
# =============================================================================
def run_training_v2():
    """
    High-speed training pipeline using pre-sliced NumPy dataset.

    Step 1: Pre-slice BraTS volumes to disk (skipped if already done).
    Step 2: Train Attention U-Net++ for 50 epochs on pre-sliced data.
    """
    def find_brats_root():
        for r, d, f in os.walk("/kaggle/input"):
            if "BraTS20_Training_001" in d:
                return r
        return None

    root = find_brats_root()
    if not root:
        print("[ERROR] BraTS 2020 dataset not found. Please add it as a Kaggle input.")
        return

    slice_dir = "/kaggle/working/fast_slices"
    if not os.path.exists(slice_dir) or len(os.listdir(slice_dir)) < 100:
        preprocess_volumes_to_slices(root, slice_dir)

    all_slices = sorted(glob.glob(f"{slice_dir}/*_img.npy"))
    random.shuffle(all_slices)
    split = int(len(all_slices) * 0.8)
    train_loader = DataLoader(PreSlicedDataset(all_slices[:split]),
                              batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(PreSlicedDataset(all_slices[split:]),
                              batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionUNetPlusPlus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    best_dice = 0.0
    metrics_csv = "/kaggle/working/training_metrics_v2.csv"

    for epoch in range(50):
        model.train()
        for img, msk in tqdm(train_loader, desc=f"Epoch {epoch+1}/50 [Train]"):
            img, msk = img.to(device), msk.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(img), msk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_metrics = {"WT": 0, "TC": 0, "ET": 0}
        with torch.no_grad():
            for img, msk in val_loader:
                img, msk = img.to(device), msk.to(device)
                reg = dice_coeff_regional(model(img), msk)
                for k in v_metrics:
                    v_metrics[k] += reg[k]

        n_val = len(val_loader)
        avg_dsc = (v_metrics["WT"] + v_metrics["TC"] + v_metrics["ET"]) / (3 * n_val)
        scheduler.step(avg_dsc)

        print(f"  Epoch {epoch+1} | DSC-WT: {v_metrics['WT']/n_val:.4f} | "
              f"DSC-TC: {v_metrics['TC']/n_val:.4f} | DSC-ET: {v_metrics['ET']/n_val:.4f} | Avg: {avg_dsc:.4f}")

        if avg_dsc > best_dice:
            best_dice = avg_dsc
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'best_dice': best_dice}, "/kaggle/working/best_checkpoint_v2.pth")
            print(f"  ⭐ New Best DSC: {best_dice:.4f}")

        pd.DataFrame([{'epoch': epoch+1, 'dsc_wt': v_metrics['WT']/n_val,
                        'dsc_tc': v_metrics['TC']/n_val, 'dsc_et': v_metrics['ET']/n_val,
                        'avg_dsc': avg_dsc}]).to_csv(
            metrics_csv, mode='a', header=not os.path.exists(metrics_csv), index=False)


if __name__ == "__main__":
    run_training_v2()
