import os, glob, torch, random, torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import pandas as pd

"""
PHD RESEARCH MASTER SCRIPT V2: HIGH-SPEED CLINICAL PIPELINE
----------------------------------------------------------
Optimized for Kaggle Commits: Pre-slices data for 10x faster training.
Includes: Attention UNet++, Z-Score, Dice-Focal Loss, and Regional Metrics.
"""

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ==========================================
# 1. FAST-SLICER ENGINE
# ==========================================
def slice_dataset(root_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_p = sorted([os.path.join(root_path, d) for d in os.listdir(root_path) if "BraTS20_Training_" in d])
    print(f"🔪 Slicing {len(all_p)} patients...")
    for p in tqdm(all_p):
        pid = os.path.basename(p); seg_file = f"{p}/{pid}_seg.nii"
        if not os.path.exists(seg_file): continue
        seg_data = nib.load(seg_file).get_fdata()
        tumor_indices = np.where(np.sum(seg_data, axis=(0,1)) > 50)[0]
        for s in tumor_indices:
            channels = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                img = nib.load(f"{p}/{pid}_{mod}.nii").get_fdata()[:,:,s]
                mask = img > 0
                if np.any(mask): img[mask] = (img[mask] - img[mask].mean()) / (img[mask].std() + 1e-8)
                channels.append(img)
            np.save(f"{output_dir}/{pid}_s{s}_img.npy", np.stack(channels, 0).astype(np.float32))
            np.save(f"{output_dir}/{pid}_s{s}_msk.npy", np.stack([(seg_data[:,:,s]>0), (seg_data[:,:,s]==1)|(seg_data[:,:,s]==4), (seg_data[:,:,s]==4)], 0).astype(np.uint8))

# ==========================================
# 2. ARCHITECTURE
# ==========================================
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_l = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x): return x * self.psi(self.relu(self.W_g(g) + self.W_l(x)))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
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
# 3. TRAINING & METRICS
# ==========================================
class FastDataset(Dataset):
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return torch.from_numpy(np.load(self.files[idx])).float(), torch.from_numpy(np.load(self.files[idx].replace('_img.npy', '_msk.npy'))).float()

def dice_coeff_regional(output, target):
    p = (torch.sigmoid(output) > 0.5).float()
    res = {}
    for i, name in enumerate(["WT", "TC", "ET"]):
        inter = (p[:,i]*target[:,i]).sum(dim=(1,2)); union = p[:,i].sum(dim=(1,2)) + target[:,i].sum(dim=(1,2))
        res[name] = ((2.*inter+1e-5)/(union+1e-5)).mean().item()
    return res

def run_v2():
    def find_data():
        for r, d, f in os.walk("/kaggle/input"):
            if "BraTS20_Training_001" in d: return r
        return None
    root = find_data()
    if not root: return
    
    slice_dir = "/kaggle/working/fast_slices"
    if not os.path.exists(slice_dir) or len(os.listdir(slice_dir)) < 100: slice_dataset(root, slice_dir)
    
    all_s = sorted(glob.glob(f"{slice_dir}/*_img.npy"))
    random.shuffle(all_s); split = int(len(all_s)*0.8); train_s, val_s = all_s[:split], all_s[split:]
    train_loader = DataLoader(FastDataset(train_s), batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(FastDataset(val_s), batch_size=32)

    model = AttentionUNetPlusPlus().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_dice = 0.0

    def criterion(out, target):
        probs = torch.sigmoid(out)
        inter = (probs*target).sum(dim=(2,3)); union = probs.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice_l = 1 - (2.*inter+1e-5)/(union+1e-5)
        bce = nn.functional.binary_cross_entropy_with_logits(out, target, reduction='none')
        pt = torch.exp(-bce); focal_l = (1-pt)**2 * bce
        return dice_l.mean() + focal_l.mean()

    for epoch in range(50):
        model.train(); t_loss = 0; t_metrics = {"WT":0, "TC":0, "ET":0}
        for img, msk in tqdm(train_loader, desc=f"E{epoch+1}"):
            img, msk = img.to("cuda"), msk.to("cuda")
            optimizer.zero_grad(); out = model(img); loss = criterion(out, msk); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); t_loss += loss.item()
        
        model.eval(); v_metrics = {"WT":0, "TC":0, "ET":0}
        with torch.no_grad():
            for img, msk in val_loader:
                img, msk = img.to("cuda"), msk.to("cuda"); out = model(img)
                reg = dice_coeff_regional(out, msk)
                for k in v_metrics: v_metrics[k] += reg[k]
        
        avg_v_dice = (v_metrics["WT"] + v_metrics["TC"] + v_metrics["ET"]) / (3 * len(val_loader))
        scheduler.step(avg_v_dice)
        print(f"📊 E{epoch+1} | Val WT: {v_metrics['WT']/len(val_loader):.4f} | TC: {v_metrics['TC']/len(val_loader):.4f} | ET: {v_metrics['ET']/len(val_loader):.4f}")
        
        if avg_v_dice > best_dice:
            best_dice = avg_v_dice; torch.save(model.state_dict(), "phd_v2_best.pth")
            print(f"⭐ New Best: {best_dice:.4f}")
        
        log_data = {'epoch':epoch+1, 'val_wt':v_metrics['WT']/len(val_loader), 'val_tc':v_metrics['TC']/len(val_loader), 'val_et':v_metrics['ET']/len(val_loader), 'avg_dice':avg_v_dice}
        pd.DataFrame([log_data]).to_csv("research_v2_log.csv", mode='a', header=not os.path.exists("research_v2_log.csv"), index=False)

if __name__ == "__main__":
    run_v2()
