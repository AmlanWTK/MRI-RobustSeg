import os
import torch
import numpy as np
import nibabel as nib
from tqdm.auto import tqdm
import pandas as pd

# IMPORTANT: Ensure these paths match your Kaggle environment
ROOT_DIR = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
CHECKPOINT_PATH = "/kaggle/input/datasets/amlan21s/latest/phd_best_checkpoint.pth"

# Model definition (must match Attention U-Net++)
import torch.nn as nn

class AttentionGate(nn.Module):
    """Attention Gate using W_g/W_l naming — matches the latest checkpoint exactly."""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_l  = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi  = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_l(x)))

class ConvBlock(nn.Module):
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


def z_score_normalize_slice(img_2d):
    """Brain-masked Z-score normalization on a single 2D slice — matches training exactly."""
    mask = img_2d > 0
    if not np.any(mask):
        return img_2d
    out = np.zeros_like(img_2d, dtype=np.float32)
    out[mask] = (img_2d[mask] - img_2d[mask].mean()) / (img_2d[mask].std() + 1e-8)
    return out


def compute_extended_metrics(pred, target):
    """Compute Dice, IoU, Sensitivity, and Specificity for a 3D volume."""
    pred   = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    tp = np.sum((pred == 1) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))

    dice        = (2. * tp + 1e-5) / (2. * tp + fp + fn + 1e-5)
    iou         = (tp + 1e-5)      / (tp + fp + fn + 1e-5)
    sensitivity = (tp + 1e-5)      / (tp + fn + 1e-5)
    specificity = (tn + 1e-5)      / (tn + fp + 1e-5)
    return dice, iou, sensitivity, specificity


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────────
    model = AttentionUNetPlusPlus().to(device)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)   # W_g/W_l matches latest checkpoint
    model.eval()
    print("✅ Checkpoint loaded successfully.")

    # ── Patient split (identical seed to training) ──────────────────────────────
    import random
    all_patients = sorted([d for d in os.listdir(ROOT_DIR) if "BraTS20_Training_" in d])
    random.seed(42)
    random.shuffle(all_patients)
    val_patients = all_patients[int(len(all_patients) * 0.8):]
    print(f"Evaluating {len(val_patients)} validation subjects...\n")

    results = {r: {"dice": [], "iou": [], "sens": [], "spec": []}
               for r in ["WT", "TC", "ET"]}

    for pid in tqdm(val_patients):
        p_path = os.path.join(ROOT_DIR, pid)
        try:
            vols = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                path = f"{p_path}/{pid}_{mod}.nii"
                if not os.path.exists(path): path += ".gz"
                vols.append(nib.load(path).get_fdata())   # (H, W, D)

            seg_path = f"{p_path}/{pid}_seg.nii"
            if not os.path.exists(seg_path): seg_path += ".gz"
            seg = nib.load(seg_path).get_fdata()

            D = vols[0].shape[2]
            pred_WT = np.zeros(seg.shape, dtype=np.uint8)
            pred_TC = np.zeros(seg.shape, dtype=np.uint8)
            pred_ET = np.zeros(seg.shape, dtype=np.uint8)

            with torch.no_grad():
                for s in range(D):
                    # ── Slice-level Z-score — identical to training ──────────
                    channels = np.stack(
                        [z_score_normalize_slice(v[:, :, s]) for v in vols], axis=0
                    ).astype(np.float32)                               # (4, H, W)

                    if np.all(channels == 0):
                        continue

                    tensor = torch.from_numpy(channels).unsqueeze(0).to(device)  # (1,4,H,W)
                    out    = torch.sigmoid(model(tensor)).cpu().numpy()[0]        # (3,H,W)
                    pred_WT[:, :, s] = (out[0] > 0.5)
                    pred_TC[:, :, s] = (out[1] > 0.5)
                    pred_ET[:, :, s] = (out[2] > 0.5)

            gt_WT = (seg > 0).astype(np.uint8)
            gt_TC = ((seg == 1) | (seg == 4)).astype(np.uint8)
            gt_ET = (seg == 4).astype(np.uint8)

            for name, gt, pr in [("WT", gt_WT, pred_WT),
                                  ("TC", gt_TC, pred_TC),
                                  ("ET", gt_ET, pred_ET)]:
                d, i, se, sp = compute_extended_metrics(pr, gt)
                results[name]["dice"].append(d)
                results[name]["iou"].append(i)
                results[name]["sens"].append(se)
                results[name]["spec"].append(sp)

        except Exception as e:
            print(f"  ⚠️  Skipping {pid}: {e}")
            continue

    # ── Final output ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("   EXTENDED CLINICAL METRICS — Proposed Framework (Final Run)")
    print("="*65)
    print(f"{'Region':<8} | {'DSC':>7} | {'IoU':>7} | {'Sensitivity':>12} | {'Specificity':>12}")
    print("-"*65)
    for name in ["WT", "TC", "ET"]:
        d  = np.mean(results[name]["dice"])
        i  = np.mean(results[name]["iou"])
        se = np.mean(results[name]["sens"])
        sp = np.mean(results[name]["spec"])
        print(f"{name:<8} | {d:>7.4f} | {i:>7.4f} | {se:>12.4f} | {sp:>12.4f}")
    print("="*65)


if __name__ == "__main__":
    main()
