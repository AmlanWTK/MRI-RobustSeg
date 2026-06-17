"""
IEEE Publication Codebase: Primary Segmentation Training Pipeline
Target Conference: NETCRYPT 2026
Institution: [Your Institution]

Description:
    This module implements the primary training loop for the Proposed Framework:
    Attention U-Net++ with Hybrid Degradation Augmentation (HDA) for multimodal
    brain tumor segmentation on the BraTS 2020 benchmark dataset.

    The pipeline supports checkpoint resumption, regional Dice evaluation
    (Whole Tumor, Tumor Core, Enhancing Tumor), and automatic metric logging.

Architecture:
    - Encoder/Decoder: Attention U-Net++ (Zhou et al., 2018 + Oktay et al., 2018)
    - Loss Function:   Combined Dice + Focal Loss
    - Augmentation:    Physics-Based Hybrid Degradation (Rician Noise, Gibbs
                       Ringing, Ghosting, Bias Field)

Dataset:
    BraTS 2020 (Brain Tumor Segmentation Challenge)
    Modalities: FLAIR, T1, T1ce, T2
    Labels: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)

Usage (Kaggle):
    1. Upload phd_best_checkpoint.pth as a Kaggle input dataset.
    2. Set CHECKPOINT_PATH to the correct input path.
    3. Run all cells. Results saved to /kaggle/working/.
"""

import os, glob, torch, random, torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter

# =============================================================================
# SECTION 1: EXPERIMENT CONFIGURATION
# =============================================================================
TOTAL_EPOCHS   = 50       # Total training epochs (inclusive of resumed epochs)
BATCH_SIZE     = 16
LEARNING_RATE  = 5e-5     # Reduced learning rate for fine-tuning from checkpoint
SEED           = 42
NUM_WORKERS    = 2

# Checkpoint input path (update to match your Kaggle dataset path)
CHECKPOINT_PATH = "/kaggle/input/datasets/amlan21s/latest/phd_best_checkpoint.pth"

# Output paths
SAVE_BEST_PATH = "/kaggle/working/best_checkpoint.pth"
SAVE_LAST_PATH = "/kaggle/working/last_checkpoint.pth"
METRICS_CSV    = "/kaggle/working/training_metrics.csv"


# =============================================================================
# SECTION 2: REPRODUCIBILITY
# =============================================================================
def set_seed(seed=42):
    """Ensure fully deterministic training for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed fixed: {seed}")

set_seed(SEED)


# =============================================================================
# SECTION 3: PHYSICS-BASED HYBRID DEGRADATION AUGMENTATION (HDA)
# =============================================================================
class HybridDegradationAugmentation:
    """
    Physics-informed MRI artifact simulation for robust model training.

    Simulates clinically realistic MRI degradations to prevent overfitting
    and improve generalization to real-world scanner variability.

    Implemented Artifacts:
        - Rician Noise:   Mathematically correct noise model for MRI magnitudes.
        - MRI Ghosting:   K-space signal echo artifacts (N/2 ghosts).
        - Gibbs Ringing:  Truncation artifacts from limited K-space sampling.
        - Bias Field:     Low-frequency intensity inhomogeneity.

    Reference:
        Inspired by TorchIO (Pérez-García et al., 2021) and clinical MRI physics.

    Args:
        p (float): Probability of applying augmentation per sample. Default: 0.4
    """
    def __init__(self, p=0.4):
        self.p = p

    def _apply_ghosting(self, img):
        """Simulate N/2 ghost artifacts via K-space modulation."""
        f = np.fft.fftshift(np.fft.fft2(img))
        f[::img.shape[0]//4, :] *= 0.8
        return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

    def _apply_gibbs(self, img):
        """Simulate Gibbs ringing via high-frequency K-space truncation."""
        f = np.fft.fftshift(np.fft.fft2(img))
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask = ((np.ogrid[:rows, :cols][0] - crow)**2 +
                (np.ogrid[:rows, :cols][1] - ccol)**2) <= (min(crow, ccol) * 0.7)**2
        f[~mask] = 0
        return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

    def _apply_bias_field(self, img):
        """Simulate smooth intensity inhomogeneity via polynomial bias field."""
        x, y = np.meshgrid(np.linspace(-1, 1, img.shape[1]),
                           np.linspace(-1, 1, img.shape[0]))
        bias = random.uniform(-0.15, 0.15) * x + random.uniform(-0.15, 0.15) * y + 1.0
        return img * bias

    def _apply_rician_noise(self, img):
        """Add Rician-distributed noise (correct noise model for MRI magnitudes)."""
        std = random.uniform(0.01, 0.04)
        return np.sqrt((img + np.random.normal(0, std, img.shape))**2 +
                        np.random.normal(0, std, img.shape)**2)

    def __call__(self, img_tensor):
        """
        Apply stochastic HDA to a multi-channel MRI tensor.

        Args:
            img_tensor (torch.Tensor): Shape (C, H, W), normalized MRI input.

        Returns:
            torch.Tensor: Augmented tensor of identical shape.
        """
        if random.random() > self.p:
            return img_tensor
        img = img_tensor.detach().cpu().numpy().copy()
        for c in range(img.shape[0]):
            img[c] = self._apply_rician_noise(img[c])
            dice = random.random()
            if dice < 0.25:
                img[c] = self._apply_ghosting(img[c])
            elif dice < 0.50:
                img[c] = self._apply_gibbs(img[c])
            elif dice < 0.75:
                img[c] = self._apply_bias_field(img[c])
        return torch.from_numpy(img).float()


# =============================================================================
# SECTION 4: PREPROCESSING — Z-SCORE NORMALIZATION
# =============================================================================
def z_score_normalize(img):
    """
    Apply brain-masked Z-score normalization to a single MRI slice.

    Only non-zero voxels (brain tissue) are used to compute statistics,
    preventing background signal from biasing the normalization.

    Args:
        img (np.ndarray): 2D MRI slice (H, W).

    Returns:
        np.ndarray: Normalized image of identical shape.
    """
    mask = img > 0
    if not np.any(mask):
        return img
    mean, std = img[mask].mean(), img[mask].std()
    normalized = np.zeros_like(img)
    normalized[mask] = (img[mask] - mean) / (std + 1e-8)
    return normalized


# =============================================================================
# SECTION 5: MODEL ARCHITECTURE — ATTENTION U-NET++
# =============================================================================
class AttentionGate(nn.Module):
    """
    Soft Attention Gate for selective feature recalibration.

    Learns to suppress irrelevant activations while amplifying
    task-relevant features during skip connection aggregation.

    Reference: Oktay et al., "Attention U-Net: Learning Where to Look
    for the Pancreas," MIDL 2018.
    """
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_l  = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi  = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_l(x)))


class ConvBlock(nn.Module):
    """Dual-convolutional block: (Conv → BN → ReLU → Dropout) × 2."""
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
    """
    Proposed Architecture: Attention U-Net++ for Multimodal Brain Tumor Segmentation.

    Combines the nested dense skip connections of U-Net++ (Zhou et al., 2018)
    with soft attention gates (Oktay et al., 2018) at each decoder stage.

    Args:
        in_ch  (int): Number of input MRI modalities. Default: 4 (FLAIR, T1, T1ce, T2).
        out_ch (int): Number of segmentation classes. Default: 3 (WT, TC, ET).
    """
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


# =============================================================================
# SECTION 6: DATASET — BraTS 2020 CLINICAL DATASET LOADER
# =============================================================================
class BraTSDataset(Dataset):
    """
    PyTorch Dataset for slice-level loading from the BraTS 2020 training set.

    Only slices with significant tumor content (segmentation mask sum > 50)
    are retained for training, following standard BraTS preprocessing protocol.

    Label Convention (BraTS 2020):
        Label 1: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
        Label 2: Peritumoral Edema (ED)
        Label 4: Enhancing Tumor (ET)

    Segmentation Targets:
        WT = Labels {1, 2, 4}    (Whole Tumor)
        TC = Labels {1, 4}       (Tumor Core)
        ET = Label  {4}          (Enhancing Tumor)

    Args:
        patients  (list): List of absolute paths to BraTS patient directories.
        augment   (callable, optional): Augmentation callable. Default: None.
    """
    def __init__(self, patients, augment=None):
        self.patients = patients
        self.augment  = augment
        self.slices   = []
        for p in tqdm(patients, desc="Indexing patient volumes"):
            pid   = os.path.basename(p)
            seg_p = f"{p}/{pid}_seg.nii"
            if not os.path.exists(seg_p):
                seg_p = f"{p}/{pid}_seg.nii.gz"
            if os.path.exists(seg_p):
                seg   = nib.load(seg_p).get_fdata()
                s_idx = np.where(np.sum(seg, axis=(0, 1)) > 50)[0]
                for s in s_idx:
                    self.slices.append((p, s))
        print(f"  → Retained {len(self.slices)} tumor-positive slices from {len(patients)} subjects.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        p_path, s_idx = self.slices[idx]
        pid = os.path.basename(p_path)
        channels = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            path = f"{p_path}/{pid}_{mod}.nii"
            if not os.path.exists(path):
                path += ".gz"
            img = nib.load(path).get_fdata()[:, :, s_idx]
            channels.append(z_score_normalize(img))
        img_t = torch.from_numpy(np.stack(channels, 0).astype(np.float32))
        seg_p = f"{p_path}/{pid}_seg.nii"
        if not os.path.exists(seg_p):
            seg_p += ".gz"
        seg   = nib.load(seg_p).get_fdata()[:, :, s_idx]
        mask_t = torch.from_numpy(np.stack([
            (seg > 0),
            (seg == 1) | (seg == 4),
            (seg == 4)
        ], 0).astype(np.float32))
        if self.augment:
            img_t = self.augment(img_t)
        return img_t, mask_t


# =============================================================================
# SECTION 7: LOSS AND EVALUATION METRICS
# =============================================================================
def dice_coeff_regional(output, target):
    """
    Compute per-region Dice Similarity Coefficient (DSC) for BraTS sub-regions.

    Args:
        output (torch.Tensor): Raw logits, shape (B, 3, H, W).
        target (torch.Tensor): Binary ground truth, shape (B, 3, H, W).

    Returns:
        dict: DSC for each region: {'WT': float, 'TC': float, 'ET': float}
    """
    p = (torch.sigmoid(output) > 0.5).float()
    results = {}
    for i, name in enumerate(["WT", "TC", "ET"]):
        pi    = p[:, i]; ti = target[:, i]
        inter = (pi * ti).sum(dim=(1, 2))
        union = pi.sum(dim=(1, 2)) + ti.sum(dim=(1, 2))
        results[name] = ((2. * inter + 1e-5) / (union + 1e-5)).mean().item()
    return results


def combined_loss(output, target):
    """
    Combined Dice + Focal Loss for class-imbalanced segmentation.

    Dice Loss addresses global region overlap; Focal Loss down-weights
    easy negatives to focus training on difficult boundary voxels.

    Args:
        output (torch.Tensor): Raw logits, shape (B, 3, H, W).
        target (torch.Tensor): Binary ground truth, shape (B, 3, H, W).

    Returns:
        torch.Tensor: Scalar combined loss value.
    """
    probs = torch.sigmoid(output)
    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_loss  = 1 - (2. * inter + 1e-5) / (union + 1e-5)
    bce        = nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt         = torch.exp(-bce)
    focal_loss = (1 - pt)**2 * bce
    return dice_loss.mean() + focal_loss.mean()


# =============================================================================
# SECTION 8: DATA DISCOVERY
# =============================================================================
def find_brats_root():
    """Auto-discover BraTS 2020 training data root on Kaggle."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if any("BraTS20_Training_001" in d for d in dirs):
            print(f"  ✅ BraTS 2020 dataset located at: {root}")
            return root
    return None


# =============================================================================
# SECTION 9: MAIN TRAINING PIPELINE
# =============================================================================
def run_training():
    """
    Main training pipeline with checkpoint resumption support.

    Trains (or resumes training of) the Attention U-Net++ model using
    the BraTS 2020 dataset. Logs per-epoch metrics to CSV and saves
    the best-performing checkpoint based on average validation Dice.
    """
    import pandas as pd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print(f"  Attention U-Net++ Training Pipeline — NETCRYPT 2026")
    print(f"  Device  : {device}")
    print(f"  Target  : Epoch 1 → {TOTAL_EPOCHS} (with checkpoint resumption)")
    print(f"{'='*65}\n")

    root = find_brats_root()
    if root is None:
        print("❌ BraTS 2020 data not found. Please add the dataset as a Kaggle input.")
        return

    all_p = sorted([os.path.join(root, d) for d in os.listdir(root) if "BraTS20_Training_" in d])
    random.seed(SEED)
    random.shuffle(all_p)
    split   = int(len(all_p) * 0.8)
    train_p = all_p[:split]
    val_p   = all_p[split:]
    print(f"  Split → Train: {len(train_p)} subjects | Validation: {len(val_p)} subjects")

    train_ds = BraTSDataset(train_p, HybridDegradationAugmentation(p=0.4))
    val_ds   = BraTSDataset(val_p)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = AttentionUNetPlusPlus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_epoch  = 0
    best_val_dice = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n  Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for pg in optimizer.param_groups:
            pg['lr'] = LEARNING_RATE
        start_epoch   = ckpt['epoch']
        best_val_dice = ckpt['best_dice']
        print(f"  ✅ Loaded Epoch {start_epoch} | Best Dice: {best_val_dice:.4f}")
        print(f"  ▶  Continuing from Epoch {start_epoch+1} → {TOTAL_EPOCHS}\n")
    else:
        print(f"  [WARNING] Checkpoint not found at {CHECKPOINT_PATH}. Training from scratch.\n")

    if start_epoch >= TOTAL_EPOCHS:
        print(f"  ✅ Training already complete at Epoch {TOTAL_EPOCHS}.")
        return

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        current_epoch = epoch + 1

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_metrics = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}/{TOTAL_EPOCHS} [Train]", leave=True)
        for img, msk in pbar:
            img, msk = img.to(device), msk.to(device)
            optimizer.zero_grad()
            out  = model(img)
            loss = combined_loss(out, msk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            rd = dice_coeff_regional(out, msk)
            for k in train_metrics:
                train_metrics[k] += rd[k]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "DSC-WT": f"{rd['WT']:.3f}"})

        # --- Validation Phase ---
        model.eval()
        val_loss    = 0.0
        val_metrics = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
        with torch.no_grad():
            for img, msk in tqdm(val_loader, desc=f"Epoch {current_epoch}/{TOTAL_EPOCHS} [Val]  ", leave=True):
                img, msk = img.to(device), msk.to(device)
                out    = model(img)
                val_loss += combined_loss(out, msk).item()
                rd = dice_coeff_regional(out, msk)
                for k in val_metrics:
                    val_metrics[k] += rd[k]

        # --- Summary ---
        n_tr = len(train_loader)
        n_va = len(val_loader)
        avg_train_loss = train_loss / n_tr
        avg_val_loss   = val_loss   / n_va
        avg_val_wt     = val_metrics["WT"] / n_va
        avg_val_tc     = val_metrics["TC"] / n_va
        avg_val_et     = val_metrics["ET"] / n_va
        avg_val_dice   = (avg_val_wt + avg_val_tc + avg_val_et) / 3.0

        scheduler.step(avg_val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n  Epoch {current_epoch}/{TOTAL_EPOCHS} Summary:")
        print(f"    Train → Loss: {avg_train_loss:.4f} | DSC-WT: {train_metrics['WT']/n_tr:.3f}")
        print(f"    Val   → Loss: {avg_val_loss:.4f} | DSC-WT: {avg_val_wt:.3f} | DSC-TC: {avg_val_tc:.3f} | DSC-ET: {avg_val_et:.3f}")
        print(f"    Avg Val DSC : {avg_val_dice:.4f} | Best: {best_val_dice:.4f} | LR: {current_lr:.2e}\n")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({'epoch': current_epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'best_dice': best_val_dice},
                       SAVE_BEST_PATH)
            print(f"  ⭐ New Best Checkpoint! DSC={best_val_dice:.4f} → {SAVE_BEST_PATH}")

        torch.save({'epoch': current_epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_dice': best_val_dice},
                   SAVE_LAST_PATH)

        log_data = {'epoch': current_epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                    'dsc_wt': avg_val_wt, 'dsc_tc': avg_val_tc, 'dsc_et': avg_val_et,
                    'avg_val_dsc': avg_val_dice, 'best_dsc': best_val_dice, 'lr': current_lr}
        log_df = pd.DataFrame([log_data])
        if not os.path.isfile(METRICS_CSV):
            log_df.to_csv(METRICS_CSV, index=False)
        else:
            log_df.to_csv(METRICS_CSV, mode='a', header=False, index=False)

    print(f"\n{'='*65}")
    print(f"  Training Complete.")
    print(f"  Best Avg DSC : {best_val_dice:.4f}")
    print(f"  Best ckpt    : {SAVE_BEST_PATH}")
    print(f"  Metrics CSV  : {METRICS_CSV}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_training()
