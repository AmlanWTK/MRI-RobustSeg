import os, torch, nibabel as nib, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn

# ==========================================
# 1. THE MODEL ARCHITECTURE (MUST MATCH MASTER SCRIPT)
# ==========================================
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_l = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
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
# 2. VISUALIZATION ENGINE
# ==========================================
def generate_clinical_figure(patient_path, checkpoint_path):
    print(f"🚀 Visualizing Patient: {os.path.basename(patient_path)}")
    
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionUNetPlusPlus().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # 2. Load MRI Modalities
    pid = os.path.basename(patient_path)
    data = {}
    for mod in ['flair', 't1', 't1ce', 't2']:
        img_path = f"{patient_path}/{pid}_{mod}.nii"
        if os.path.exists(img_path + ".gz"): img_path += ".gz"
        data[mod] = nib.load(img_path).get_fdata()
    
    seg_path = f"{patient_path}/{pid}_seg.nii"
    if os.path.exists(seg_path + ".gz"): seg_path += ".gz"
    seg = nib.load(seg_path).get_fdata()

    # 3. Find the best slice (center of tumor)
    tumor_mass = np.sum(seg, axis=(0,1))
    best_slice = np.argmax(tumor_mass)
    print(f"📍 Selecting Slice: {best_slice}")

    # 4. Prepare Model Input
    input_stack = []
    for mod in ['flair', 't1', 't1ce', 't2']:
        slice_img = data[mod][:, :, best_slice]
        # Z-score normalization
        mask = slice_img > 0
        if np.any(mask):
            slice_img[mask] = (slice_img[mask] - slice_img[mask].mean()) / (slice_img[mask].std() + 1e-8)
        input_stack.append(slice_img)
    
    input_tensor = torch.from_numpy(np.stack(input_stack, 0)).float().unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]
    
    # Thresholding
    pred_wt = output[0] > 0.5
    pred_tc = output[1] > 0.5
    pred_et = output[2] > 0.5

    # 6. Plotting
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 4, 1)
    plt.imshow(data['flair'][:,:,best_slice], cmap='gray')
    plt.title("A: FLAIR (Input)", fontsize=15); plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(data['t1ce'][:,:,best_slice], cmap='gray')
    plt.title("B: T1ce (Contrast)", fontsize=15); plt.axis('off')

    # Panel C: Ground Truth Overlay
    gt_wt = seg[:,:,best_slice] > 0
    plt.subplot(1, 4, 3)
    plt.imshow(data['flair'][:,:,best_slice], cmap='gray')
    plt.imshow(gt_wt, alpha=0.4, cmap='Reds')
    plt.title("C: Clinical Ground Truth", fontsize=15); plt.axis('off')

    # Panel D: Prediction Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(data['flair'][:,:,best_slice], cmap='gray')
    plt.imshow(pred_wt, alpha=0.4, cmap='Greens')
    plt.title(f"D: Attention UNet++ (Our Result)", fontsize=15, color='green'); plt.axis('off')

    plt.tight_layout()
    plt.savefig("phd_paper_figure_1.png", dpi=300)
    print("✅ Figure saved as 'phd_paper_figure_1.png'")
    plt.show()

if __name__ == "__main__":
    PATIENT_PATH = r"D:\Downloads\BrainTumorDataSet\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001"
    CHECKPOINT = "phd_best_checkpoint (3).pth"
    generate_clinical_figure(PATIENT_PATH, CHECKPOINT)
