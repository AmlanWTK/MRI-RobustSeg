import os, glob, torch, random, torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter

"""
PHD RESEARCH MASTER SCRIPT: CLINICAL GRADE BRAIN TUMOR SEGMENTATION
------------------------------------------------------------------
Includes: Attention UNet++, Z-Score Normalization, Patient-Level Splitting,
          MC Dropout, Physics Augmentation, and Full Validation Loop.
"""

# ==========================================
# 0. SCIENTIFIC REPRODUCIBILITY (SEEDS)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Reproducibility Seed set to: {seed}")

set_seed(42)

# ==========================================
# 0. ADVANCED CLINICAL AUGMENTATIONS
# ==========================================
class Q1HybridDegradation:
    def __init__(self, p=0.4):
        self.p = p
    
    def _apply_ghosting(self, img):
        f = np.fft.fftshift(np.fft.fft2(img))
        f[::img.shape[0]//4, :] *= 0.8
        return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

    def _apply_gibbs(self, img):
        f = np.fft.fftshift(np.fft.fft2(img))
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        mask = (np.ogrid[:rows, :cols][0] - crow)**2 + (np.ogrid[:rows, :cols][1] - ccol)**2 <= (min(crow,ccol)*0.7)**2
        f[~mask] = 0
        return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

    def _apply_bias_field(self, img):
        x, y = np.meshgrid(np.linspace(-1, 1, img.shape[1]), np.linspace(-1, 1, img.shape[0]))
        poly = (random.uniform(-0.15, 0.15) * x + random.uniform(-0.15, 0.15) * y + 1.0)
        return img * poly

    def _apply_rician(self, img):
        std = random.uniform(0.01, 0.04)
        return np.sqrt((img + np.random.normal(0, std, img.shape))**2 + np.random.normal(0, std, img.shape)**2)

    def __call__(self, img_tensor):
        if random.random() > self.p: return img_tensor
        # SAFETY FIX: Detach and copy to avoid memory issues
        img = img_tensor.detach().cpu().numpy().copy()
        
        for c in range(img.shape[0]):
            img[c] = self._apply_rician(img[c])
            dice = random.random()
            if dice < 0.25: img[c] = self._apply_ghosting(img[c])
            elif dice < 0.50: img[c] = self._apply_gibbs(img[c])
            elif dice < 0.75: img[c] = self._apply_bias_field(img[c])
            
        return torch.from_numpy(img).float()

# ==========================================
# 1. CLINICAL NORMALIZATION & MODELS
# ==========================================
def z_score_normalize(img):
    mask = img > 0
    if not np.any(mask): return img
    mean, std = img[mask].mean(), img[mask].std()
    normalized = np.zeros_like(img)
    normalized[mask] = (img[mask] - mean) / (std + 1e-8)
    return normalized

class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_l = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.psi(self.relu(self.W_g(g) + self.W_l(x)))
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionUNetPlusPlus(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.c00 = ConvBlock(in_ch, f[0]); self.c10 = ConvBlock(f[0], f[1]); self.c20 = ConvBlock(f[1], f[2])
        self.c30 = ConvBlock(f[2], f[3]); self.c40 = ConvBlock(f[3], f[4]); self.p = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2); self.att4 = AttentionGate(f[3], f[3], f[2]); self.d4 = ConvBlock(f[3]*2, f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2); self.att3 = AttentionGate(f[2], f[2], f[1]); self.d3 = ConvBlock(f[2]*2, f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2); self.att2 = AttentionGate(f[1], f[1], f[0]); self.d2 = ConvBlock(f[1]*2, f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2); self.att1 = AttentionGate(f[0], f[0], f[0]//2); self.d1 = ConvBlock(f[0]*2, f[0])
        self.final = nn.Conv2d(f[0], out_ch, 1)
    def forward(self, x):
        x00 = self.c00(x); x10 = self.c10(self.p(x00)); x20 = self.c20(self.p(x10))
        x30 = self.c30(self.p(x20)); x40 = self.c40(self.p(x30))
        g4 = self.up4(x40); u4 = self.d4(torch.cat([g4, self.att4(g4, x30)], 1))
        g3 = self.up3(u4); u3 = self.d3(torch.cat([g3, self.att3(g3, x20)], 1))
        g2 = self.up2(u3); u2 = self.d2(torch.cat([g2, self.att2(g2, x10)], 1))
        g1 = self.up1(u2); u1 = self.d1(torch.cat([g1, self.att1(g1, x00)], 1))
        return self.final(u1)

# ==========================================
# 2. CLINICAL DATASET & METRICS
# ==========================================
class ClinicalDataset(Dataset):
    def __init__(self, patients, augment=None):
        self.patients = patients; self.augment = augment; self.slices = []
        for p in tqdm(patients, desc="Indexing"):
            pid = os.path.basename(p); seg_p = f"{p}/{pid}_seg.nii"
            if os.path.exists(seg_p):
                seg = nib.load(seg_p).get_fdata()
                s_idx = np.where(np.sum(seg, axis=(0,1)) > 50)[0]
                for s in s_idx: self.slices.append((p, s))

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        p_path, s_idx = self.slices[idx]; pid = os.path.basename(p_path)
        channels = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            img = nib.load(f"{p_path}/{pid}_{mod}.nii").get_fdata()[:,:,s_idx]
            channels.append(z_score_normalize(img))
        img_t = torch.from_numpy(np.stack(channels, 0).astype(np.float32))
        seg = nib.load(f"{p_path}/{pid}_seg.nii").get_fdata()[:,:,s_idx]
        mask_t = torch.from_numpy(np.stack([(seg>0), (seg==1)|(seg==4), (seg==4)], 0).astype(np.float32))
        if self.augment: img_t = self.augment(img_t)
        return img_t, mask_t

def dice_coeff_regional(output, target):
    """Calculates Dice for Whole Tumor (0), Tumor Core (1), and Enhancing Tumor (2)"""
    p = (torch.sigmoid(output) > 0.5).float()
    regions = ["WT", "TC", "ET"]
    results = {}
    for i, name in enumerate(regions):
        pi = p[:, i]; ti = target[:, i]
        inter = (pi * ti).sum(dim=(1, 2)); union = pi.sum(dim=(1, 2)) + ti.sum(dim=(1, 2))
        results[name] = ((2. * inter + 1e-5) / (union + 1e-5)).mean().item()
    return results

# ==========================================
# 3. MASTER TRAINING & VALIDATION LOOP
# ==========================================
def run_research():
    def find_data():
        for r, d, f in os.walk("/kaggle/input"):
            if "BraTS20_Training_001" in d: return r
        return None
    
    root = find_data()
    if not root: return
    
    all_p = sorted([os.path.join(root, d) for d in os.listdir(root) if "BraTS20_Training_" in d])
    random.shuffle(all_p)
    split = int(len(all_p) * 0.8)
    train_p, val_p = all_p[:split], all_p[split:]

    train_loader = DataLoader(ClinicalDataset(train_p, Q1HybridDegradation()), batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(ClinicalDataset(val_p), batch_size=16)

    model = AttentionUNetPlusPlus().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # LR Scheduler: Reduces LR when validation Dice stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_dice = 0.0

    def criterion(output, target):
        probs = torch.sigmoid(output)
        inter = (probs * target).sum(dim=(2,3)); union = probs.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice_loss = 1 - (2. * inter + 1e-5) / (union + 1e-5)
        bce = nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = (1 - pt)**2 * bce
        return dice_loss.mean() + focal_loss.mean()

    # ==========================================
    # 5. RESUME LOGIC (OPTIONAL)
    # ==========================================
    start_epoch = 0
    best_val_dice = 0.0
    # Point this to where your epoch 22 checkpoint is located.
    checkpoint_path = "/kaggle/input/datasets/puspenmandal/lastdataset/phd_best_checkpoint.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_dice = checkpoint['best_dice']
        print(f"✅ Loaded Epoch {start_epoch} with Best Dice: {best_val_dice:.4f}")

    for epoch in range(start_epoch, 30): # Set to 30 as requested
        # --- TRAINING ---
        model.train(); train_loss = 0; train_metrics = {"WT": 0, "TC": 0, "ET": 0}
        pbar = tqdm(train_loader, desc=f"E{epoch+1} [Train]")
        for img, msk in pbar:
            img, msk = img.to("cuda"), msk.to("cuda")
            optimizer.zero_grad(); out = model(img); loss = criterion(out, msk)
            loss.backward()
            # Gradient Clipping: Prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            reg_dice = dice_coeff_regional(out, msk)
            for k in train_metrics: train_metrics[k] += reg_dice[k]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "WT": f"{reg_dice['WT']:.2f}"})

        # --- VALIDATION ---
        model.eval(); val_loss = 0; val_metrics = {"WT": 0, "TC": 0, "ET": 0}
        with torch.no_grad():
            for img, msk in tqdm(val_loader, desc=f"E{epoch+1} [Val]"):
                img, msk = img.to("cuda"), msk.to("cuda")
                out = model(img); v_loss = criterion(out, msk)
                val_loss += v_loss.item()
                reg_dice = dice_coeff_regional(out, msk)
                for k in val_metrics: val_metrics[k] += reg_dice[k]
        
        # Calculate Averages
        avg_v_loss = val_loss / len(val_loader)
        avg_v_dice = (val_metrics["WT"] + val_metrics["TC"] + val_metrics["ET"]) / (3 * len(val_loader))
        
        # Step the Scheduler
        scheduler.step(avg_v_dice)

        print(f"📊 E{epoch+1} Summary:")
        print(f"   Train -> Loss: {train_loss/len(train_loader):.4f} | WT: {train_metrics['WT']/len(train_loader):.2f}")
        print(f"   Val   -> Loss: {avg_v_loss:.4f} | WT: {val_metrics['WT']/len(val_loader):.2f} | TC: {val_metrics['TC']/len(val_loader):.2f} | ET: {val_metrics['ET']/len(val_loader):.2f}")
        
        # Save Best Model (Full Checkpoint for Resuming)
        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_val_dice,
            }
            # Saving to a NEW file so it doesn't overwrite your Epoch 22 version
            torch.save(checkpoint, "phd_best_checkpoint_run2.pth")
            print(f"⭐ New Best Model! Avg Dice: {best_val_dice:.4f}")

        # Save Metrics to CSV
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': avg_v_loss,
            'train_wt_dice': train_metrics['WT'] / len(train_loader),
            'val_wt_dice': val_metrics['WT'] / len(val_loader),
            'val_tc_dice': val_metrics['TC'] / len(val_loader),
            'val_et_dice': val_metrics['ET'] / len(val_loader),
            'best_dice_so_far': best_val_dice
        }
        
        import pandas as pd
        log_df = pd.DataFrame([log_data])
        log_file = "research_metrics_run2.csv"
        if not os.path.isfile(log_file):
            log_df.to_csv(log_file, index=False)
        else:
            log_df.to_csv(log_file, mode='a', header=False, index=False)

        torch.save(model.state_dict(), "phd_last_model_run2.pth")

if __name__ == "__main__":
    run_research()
