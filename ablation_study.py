"""
IEEE Publication Codebase: Ablation Study Framework
Target Conference: NETCRYPT 
Description: This module performs an ablation study to systematically evaluate 
the contribution of architectural and physics-based augmentation components 
on multimodal brain tumor segmentation performance.

Evaluated Architectures:
1. Standard U-Net (Baseline)
2. U-Net++ (Nested Architecture)
3. Attention U-Net++ (Without Physics-Informed Augmentation)
4. Proposed Framework (Attention U-Net++ with Hybrid Degradation Augmentation)
"""

import os, random, torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion

SEED = 42
ABLATION_EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4
NUM_WORKERS = 0  # 0 prevents DataLoader deadlocks on Kaggle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed(s); torch.backends.cudnn.deterministic = True

set_seed(SEED)

# ====== NORMALIZATION ======
def z_score_normalize(img):
    mask = img > 0
    if not np.any(mask): return img
    m, s = img[mask].mean(), img[mask].std()
    o = np.zeros_like(img); o[mask] = (img[mask] - m) / (s + 1e-8)
    return o

# ====== AUGMENTATION (only for full model) ======
class Q1HybridDegradation:
    def __init__(self, p=0.4): self.p = p
    def _ghosting(self, img):
        f = np.fft.fftshift(np.fft.fft2(img)); f[::img.shape[0]//4,:] *= 0.8
        return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))
    def _gibbs(self, img):
        f = np.fft.fftshift(np.fft.fft2(img)); r,c = img.shape; cr,cc = r//2,c//2
        mask = ((np.ogrid[:r,:c][0]-cr)**2+(np.ogrid[:r,:c][1]-cc)**2)<=(min(cr,cc)*0.7)**2
        f[~mask]=0; return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))
    def _bias(self, img):
        x,y = np.meshgrid(np.linspace(-1,1,img.shape[1]),np.linspace(-1,1,img.shape[0]))
        return img*(random.uniform(-0.15,0.15)*x+random.uniform(-0.15,0.15)*y+1.0)
    def _rician(self, img):
        s=random.uniform(0.01,0.04)
        return np.sqrt((img+np.random.normal(0,s,img.shape))**2+np.random.normal(0,s,img.shape)**2)
    def __call__(self, t):
        if random.random()>self.p: return t
        img=t.detach().cpu().numpy().copy()
        for c in range(img.shape[0]):
            img[c]=self._rician(img[c]); d=random.random()
            if d<0.25: img[c]=self._ghosting(img[c])
            elif d<0.5: img[c]=self._gibbs(img[c])
            elif d<0.75: img[c]=self._bias(img[c])
        return torch.from_numpy(img).float()

# ====== DATASET ======
class ClinicalDataset(Dataset):
    def __init__(self, patients, augment=None):
        self.augment = augment; self.slices = []
        for p in tqdm(patients, desc="Indexing"):
            pid = os.path.basename(p)
            sp = f"{p}/{pid}_seg.nii"
            if not os.path.exists(sp): sp += ".gz"
            if os.path.exists(sp):
                seg = nib.load(sp).get_fdata()
                for s in np.where(np.sum(seg, axis=(0,1)) > 50)[0]:
                    self.slices.append((p, s))
    def __len__(self): return len(self.slices)
    def __getitem__(self, idx):
        pp, si = self.slices[idx]; pid = os.path.basename(pp); chs = []
        for mod in ['flair','t1','t1ce','t2']:
            path = f"{pp}/{pid}_{mod}.nii"
            if not os.path.exists(path): path += ".gz"
            chs.append(z_score_normalize(nib.load(path).get_fdata()[:,:,si]))
        img = torch.from_numpy(np.stack(chs,0).astype(np.float32))
        sp = f"{pp}/{pid}_seg.nii"
        if not os.path.exists(sp): sp += ".gz"
        seg = nib.load(sp).get_fdata()[:,:,si]
        msk = torch.from_numpy(np.stack([(seg>0),(seg==1)|(seg==4),(seg==4)],0).astype(np.float32))
        if self.augment: img = self.augment(img)
        return img, msk

# ====== BUILDING BLOCKS ======
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, drop=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic,oc,3,padding=1),nn.BatchNorm2d(oc),nn.ReLU(True),
            nn.Dropout2d(drop),
            nn.Conv2d(oc,oc,3,padding=1),nn.BatchNorm2d(oc),nn.ReLU(True))
    def forward(self,x): return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, fg, fl, fi):
        super().__init__()
        self.Wg=nn.Sequential(nn.Conv2d(fg,fi,1),nn.BatchNorm2d(fi))
        self.Wl=nn.Sequential(nn.Conv2d(fl,fi,1),nn.BatchNorm2d(fi))
        self.psi=nn.Sequential(nn.Conv2d(fi,1,1),nn.BatchNorm2d(1),nn.Sigmoid())
        self.relu=nn.ReLU(True)
    def forward(self,g,x): return x*self.psi(self.relu(self.Wg(g)+self.Wl(x)))

# ====== ARCHITECTURE 1: STANDARD U-NET (BASELINE) ======
class BaselineUNet(nn.Module):
    def __init__(self, ic=4, oc=3):
        super().__init__()
        f=[32,64,128,256,512]
        self.e1=ConvBlock(ic,f[0]); self.e2=ConvBlock(f[0],f[1])
        self.e3=ConvBlock(f[1],f[2]); self.e4=ConvBlock(f[2],f[3])
        self.b=ConvBlock(f[3],f[4]); self.p=nn.MaxPool2d(2)
        self.u4=nn.ConvTranspose2d(f[4],f[3],2,stride=2); self.d4=ConvBlock(f[3]*2,f[3])
        self.u3=nn.ConvTranspose2d(f[3],f[2],2,stride=2); self.d3=ConvBlock(f[2]*2,f[2])
        self.u2=nn.ConvTranspose2d(f[2],f[1],2,stride=2); self.d2=ConvBlock(f[1]*2,f[1])
        self.u1=nn.ConvTranspose2d(f[1],f[0],2,stride=2); self.d1=ConvBlock(f[0]*2,f[0])
        self.final=nn.Conv2d(f[0],oc,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p(e1))
        e3=self.e3(self.p(e2)); e4=self.e4(self.p(e3))
        b=self.b(self.p(e4))
        d4=self.d4(torch.cat([self.u4(b),e4],1))
        d3=self.d3(torch.cat([self.u3(d4),e3],1))
        d2=self.d2(torch.cat([self.u2(d3),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.final(d1)

# ====== ARCHITECTURE 2: U-NET++ (NESTED, NO ATTENTION) ======
class UNetPlusPlus(nn.Module):
    def __init__(self, ic=4, oc=3):
        super().__init__()
        f=[32,64,128,256,512]
        self.c00=ConvBlock(ic,f[0]); self.c10=ConvBlock(f[0],f[1])
        self.c20=ConvBlock(f[1],f[2]); self.c30=ConvBlock(f[2],f[3])
        self.c40=ConvBlock(f[3],f[4]); self.p=nn.MaxPool2d(2)
        self.up4=nn.ConvTranspose2d(f[4],f[3],2,stride=2); self.d4=ConvBlock(f[3]*2,f[3])
        self.up3=nn.ConvTranspose2d(f[3],f[2],2,stride=2); self.d3=ConvBlock(f[2]*2,f[2])
        self.up2=nn.ConvTranspose2d(f[2],f[1],2,stride=2); self.d2=ConvBlock(f[1]*2,f[1])
        self.up1=nn.ConvTranspose2d(f[1],f[0],2,stride=2); self.d1=ConvBlock(f[0]*2,f[0])
        self.final=nn.Conv2d(f[0],oc,1)
    def forward(self,x):
        x00=self.c00(x); x10=self.c10(self.p(x00))
        x20=self.c20(self.p(x10)); x30=self.c30(self.p(x20))
        x40=self.c40(self.p(x30))
        g4=self.up4(x40); u4=self.d4(torch.cat([g4,x30],1))
        g3=self.up3(u4); u3=self.d3(torch.cat([g3,x20],1))
        g2=self.up2(u3); u2=self.d2(torch.cat([g2,x10],1))
        g1=self.up1(u2); u1=self.d1(torch.cat([g1,x00],1))
        return self.final(u1)

# ====== ARCHITECTURE 3: ATTENTION U-NET++ (PROPOSED ARCHITECTURE) ======
class AttentionUNetPlusPlus(nn.Module):
    def __init__(self, ic=4, oc=3):
        super().__init__()
        f=[32,64,128,256,512]
        self.c00=ConvBlock(ic,f[0]); self.c10=ConvBlock(f[0],f[1])
        self.c20=ConvBlock(f[1],f[2]); self.c30=ConvBlock(f[2],f[3])
        self.c40=ConvBlock(f[3],f[4]); self.p=nn.MaxPool2d(2)
        self.up4=nn.ConvTranspose2d(f[4],f[3],2,stride=2)
        self.att4=AttentionGate(f[3],f[3],f[2]); self.d4=ConvBlock(f[3]*2,f[3])
        self.up3=nn.ConvTranspose2d(f[3],f[2],2,stride=2)
        self.att3=AttentionGate(f[2],f[2],f[1]); self.d3=ConvBlock(f[2]*2,f[2])
        self.up2=nn.ConvTranspose2d(f[2],f[1],2,stride=2)
        self.att2=AttentionGate(f[1],f[1],f[0]); self.d2=ConvBlock(f[1]*2,f[1])
        self.up1=nn.ConvTranspose2d(f[1],f[0],2,stride=2)
        self.att1=AttentionGate(f[0],f[0],f[0]//2); self.d1=ConvBlock(f[0]*2,f[0])
        self.final=nn.Conv2d(f[0],oc,1)
    def forward(self,x):
        x00=self.c00(x); x10=self.c10(self.p(x00))
        x20=self.c20(self.p(x10)); x30=self.c30(self.p(x20))
        x40=self.c40(self.p(x30))
        g4=self.up4(x40); u4=self.d4(torch.cat([g4,self.att4(g4,x30)],1))
        g3=self.up3(u4); u3=self.d3(torch.cat([g3,self.att3(g3,x20)],1))
        g2=self.up2(u3); u2=self.d2(torch.cat([g2,self.att2(g2,x10)],1))
        g1=self.up1(u2); u1=self.d1(torch.cat([g1,self.att1(g1,x00)],1))
        return self.final(u1)

# ====== LOSS ======
def combined_loss(output, target):
    probs = torch.sigmoid(output)
    inter = (probs*target).sum(dim=(2,3)); union = probs.sum(dim=(2,3))+target.sum(dim=(2,3))
    dl = 1-(2.*inter+1e-5)/(union+1e-5)
    bce = nn.functional.binary_cross_entropy_with_logits(output,target,reduction='none')
    fl = (1-torch.exp(-bce))**2 * bce
    return dl.mean()+fl.mean()

# ====== METRICS ======
def dice_regional(output, target):
    p=(torch.sigmoid(output)>0.5).float(); r={}
    for i,n in enumerate(["WT","TC","ET"]):
        pi=p[:,i]; ti=target[:,i]
        inter=(pi*ti).sum(dim=(1,2)); union=pi.sum(dim=(1,2))+ti.sum(dim=(1,2))
        r[n]=((2.*inter+1e-5)/(union+1e-5)).mean().item()
    return r

def compute_hd95(pred, gt):
    pb, gb = pred.astype(bool), gt.astype(bool)
    if not pb.any() and not gb.any(): return 0.0
    if not pb.any() or not gb.any(): return 373.13
    ps = pb & ~binary_erosion(pb); gs = gb & ~binary_erosion(gb)
    d1 = distance_transform_edt(~gs)[ps]; d2 = distance_transform_edt(~ps)[gs]
    return float(np.percentile(np.concatenate([d1,d2]), 95))

def compute_dice_3d(p, g):
    inter = np.logical_and(p,g).sum(); union = p.sum()+g.sum()
    return (2.*inter)/union if union>0 else 1.0

# ====== 3D PATIENT EVALUATION ======
def eval_patient(ppath, model):
    pid = os.path.basename(ppath)
    sp = f"{ppath}/{pid}_seg.nii"
    if not os.path.exists(sp): sp += ".gz"
    seg3d = nib.load(sp).get_fdata(); H,W,D = seg3d.shape
    gt_wt=(seg3d>0).astype(np.uint8); gt_tc=((seg3d==1)|(seg3d==4)).astype(np.uint8)
    gt_et=(seg3d==4).astype(np.uint8)
    vols = {}
    for mod in ['flair','t1','t1ce','t2']:
        path=f"{ppath}/{pid}_{mod}.nii"
        if not os.path.exists(path): path+=".gz"
        vols[mod]=nib.load(path).get_fdata()
    pred_wt=np.zeros((H,W,D),dtype=np.uint8)
    pred_tc=np.zeros((H,W,D),dtype=np.uint8)
    pred_et=np.zeros((H,W,D),dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for s in range(D):
            chs=[z_score_normalize(vols[m][:,:,s]) for m in ['flair','t1','t1ce','t2']]
            inp=torch.from_numpy(np.stack(chs,0).astype(np.float32)).unsqueeze(0).to(DEVICE)
            out=torch.sigmoid(model(inp)).squeeze(0).cpu().numpy()
            pred_wt[:,:,s]=(out[0]>0.5); pred_tc[:,:,s]=(out[1]>0.5); pred_et[:,:,s]=(out[2]>0.5)
    res = {"patient_id": pid}
    for rn, pr, gt in [("WT",pred_wt,gt_wt),("TC",pred_tc,gt_tc),("ET",pred_et,gt_et)]:
        res[f"dice_{rn}"]=compute_dice_3d(pr.astype(bool),gt.astype(bool))
        res[f"hd95_{rn}"]=compute_hd95(pr,gt)
    return res

# ====== DATA DISCOVERY ======
def find_brats():
    for r,d,_ in os.walk("/kaggle/input"):
        if any("BraTS20_Training_001" in x for x in d): return r
    return None

# ====== TRAIN ABLATION COMPONENT ======
def train_variant(name, model, train_loader, val_loader, epochs=15, start_epoch=0):
    print(f"\n{'='*60}")
    print(f"  Evaluating Ablation Configuration: {name} (Epochs {start_epoch+1} to {epochs})")
    print(f"{'='*60}")
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_dice = 0.0
    for epoch in range(start_epoch, epochs):
        model.train(); tl=0; tm={"WT":0,"TC":0,"ET":0}
        for img,msk in tqdm(train_loader, desc=f"{name} E{epoch+1} [T]"):
            img,msk=img.to(DEVICE),msk.to(DEVICE)
            opt.zero_grad(); out=model(img); loss=combined_loss(out,msk)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tl+=loss.item(); rd=dice_regional(out,msk)
            for k in tm: tm[k]+=rd[k]
        model.eval(); vl=0; vm={"WT":0,"TC":0,"ET":0}
        with torch.no_grad():
            for img,msk in tqdm(val_loader, desc=f"{name} E{epoch+1} [V]"):
                img,msk=img.to(DEVICE),msk.to(DEVICE)
                out=model(img); vl+=combined_loss(out,msk).item()
                rd=dice_regional(out,msk)
                for k in vm: vm[k]+=rd[k]
        nt=len(train_loader); nv=len(val_loader)
        avg_dice=(vm["WT"]/nv+vm["TC"]/nv+vm["ET"]/nv)/3
        print(f"  E{epoch+1}: TrLoss={tl/nt:.4f} | ValWT={vm['WT']/nv:.3f} TC={vm['TC']/nv:.3f} ET={vm['ET']/nv:.3f} | Avg={avg_dice:.4f}")
        if avg_dice > best_dice:
            best_dice = avg_dice
            ckpt = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'best_dice': best_dice}
            torch.save(ckpt, f"/kaggle/working/ablation_{name}.pth")
            print(f"  ⭐ New best: {best_dice:.4f}")
    return model, best_dice

# ====== 3D VOLUMETRIC EVALUATION ======
def eval_variant(name, model, val_patients):
    print(f"\n🔬 Commencing 3D Volumetric Evaluation for {name} ({len(val_patients)} subjects)...")
    results = []
    for pp in tqdm(val_patients, desc=f"Eval {name}"):
        try: results.append(eval_patient(pp, model))
        except: pass
    df = pd.DataFrame(results)
    means = df[[c for c in df.columns if c != "patient_id"]].mean()
    print(f"  {name} → WT={means['dice_WT']:.4f} TC={means['dice_TC']:.4f} ET={means['dice_ET']:.4f}")
    print(f"         HD95: WT={means['hd95_WT']:.2f} TC={means['hd95_TC']:.2f} ET={means['hd95_ET']:.2f}")
    df.to_csv(f"/kaggle/working/ablation_{name}_patients.csv", index=False)
    return means

# ====== HELPER: TRAIN OR LOAD ======
def train_or_load(name, model_class, train_loader, val_loader, val_patients, epochs=15):
    """If ablation checkpoint exists, load it. Resume if not finished."""
    search_paths = [
        f"/kaggle/working/ablation_{name}.pth",
        f"/kaggle/input/datasets/amlan21s/ablation-latest/ablation_{name} (1).pth",
        f"/kaggle/input/datasets/amlan21s/ablation-latest/ablation_{name}.pth",
        f"/kaggle/input/datasets/amlan21s/ablation/ablation_{name}.pth",
        f"/kaggle/input/datasets/amlan21s/ablation1/ablation_{name}.pth",
        f"/kaggle/input/datasets/amlan21s/latest/ablation_{name}.pth"
    ]
    
    model = model_class().to(DEVICE)
    start_epoch = 0
    loaded = False
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"\n✅ Found checkpoint: {path}")
            ckpt = torch.load(path, map_location=DEVICE)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
            else:
                model.load_state_dict(ckpt)
                start_epoch = 9 # Because we know it stopped at E9!
            loaded = True
            break
    
    if start_epoch < epochs:
        if loaded: print(f"🔄 RESUMING training from Epoch {start_epoch + 1} to {epochs}")
        model, _ = train_variant(name, model, train_loader, val_loader, epochs, start_epoch)
    else:
        print(f"✅ Model {name} already finished {epochs} epochs. SKIPPING training.")
        
    results = eval_variant(name, model, val_patients)
    del model; torch.cuda.empty_cache()
    return results

# ====== MAIN ======
def run_ablation():
    root = find_brats()
    if not root: print("BraTS not found!"); return
    all_p = sorted([os.path.join(root,d) for d in os.listdir(root) if "BraTS20_Training_" in d])
    random.seed(SEED); random.shuffle(all_p)
    split = int(len(all_p)*0.8)
    train_p, val_p = all_p[:split], all_p[split:]

    train_ds_clean = ClinicalDataset(train_p)  # No augmentation
    val_ds = ClinicalDataset(val_p)
    tl_clean = DataLoader(train_ds_clean, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    all_results = {}

    # V1: Baseline UNet (no attention, no physics aug)
    # SKIPPING because we already finished this in the previous run
    # all_results["Baseline UNet"] = train_or_load("BaselineUNet", BaselineUNet, tl_clean, vl, val_p, ABLATION_EPOCHS)

    # V2: UNet++ (no attention, no physics aug)
    # SKIPPING because we already finished this in the previous run
    # all_results["UNet++"] = train_or_load("UNetPP", UNetPlusPlus, tl_clean, vl, val_p, ABLATION_EPOCHS)

    # V3: Attention UNet++ WITHOUT physics aug
    all_results["Att-UNet++ (no aug)"] = train_or_load("AttUNetPP_NoAug", AttentionUNetPlusPlus, tl_clean, vl, val_p, ABLATION_EPOCHS)

    # V4: Proposed Framework
    print("\n🔄 Initializing Proposed Framework (Attention U-Net++ with Hybrid Degradation)...")
    ckpt_path = "/kaggle/working/phd_best_checkpoint_run3.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/kaggle/input/datasets/amlan21s/latest/phd_best_checkpoint.pth"
    m4 = AttentionUNetPlusPlus().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if 'model_state_dict' in ckpt: m4.load_state_dict(ckpt['model_state_dict'])
    else: m4.load_state_dict(ckpt)
    all_results["Proposed Framework"] = eval_variant("Proposed Framework", m4, val_p)

    # ====== IEEE PUBLICATION RESULTS TABLE ======
    print(f"\n{'='*85}")
    print(f"  TABLE I: QUANTITATIVE ABLATION RESULTS (NETCRYPT)")
    print(f"{'='*85}")
    print(f"  {'Configuration':<30} {'Dice WT':>9} {'Dice TC':>9} {'Dice ET':>9} {'HD95 WT':>9} {'HD95 TC':>9} {'HD95 ET':>9}")
    print(f"  {'-'*30} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for name, m in all_results.items():
        print(f"  {name:<30} {m['dice_WT']:>9.4f} {m['dice_TC']:>9.4f} {m['dice_ET']:>9.4f} {m['hd95_WT']:>9.2f} {m['hd95_TC']:>9.2f} {m['hd95_ET']:>9.2f}")
    print(f"{'='*85}")

    # Save table
    rows = []
    for name, m in all_results.items():
        row = {"Model": name}
        for k in m.index: row[k] = m[k]
        rows.append(row)
    pd.DataFrame(rows).to_csv("/kaggle/working/ablation_table.csv", index=False)
    print("✅ Saved → /kaggle/working/ablation_table.csv")

run_ablation()

