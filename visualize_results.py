import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import UNetPlusPlus
from dataset import BrainTumorTorchDataset, load_splits
from config import Config
import os

"""
RESEARCH VISUALIZATION ENGINE
----------------------------
Use this to generate the figures for your 'Results' section.
It will create high-quality side-by-side comparisons of:
[FLAIR MRI] | [Ground Truth] | [UNet++ Prediction]
"""

def generate_paper_figures():
    # 1. Setup
    device = torch.device(Config.DEVICE)
    MODEL_PATH = "final_research_model.pth" # The file you download from Kaggle
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found. Please download it from Kaggle first!")
        return

    # 2. Load Model
    # Note: Kaggle script used 3 output channels (WT, TC, ET)
    model = UNetPlusPlus(in_ch=4, out_ch=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Research Model Loaded: {MODEL_PATH}")

    # 3. Prepare Test Samples
    _, _, test_files = load_splits(Config.SPLIT_DIR)
    test_ds = BrainTumorTorchDataset(test_files)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    if not os.path.exists("research_gallery"):
        os.makedirs("research_gallery")

    print("🖼️ Generating Gallery...")
    count = 0
    with torch.no_grad():
        for images, masks, _ in test_loader:
            if count >= 10: break # Generate 10 examples
            
            # Skip empty slices for the gallery to show tumors
            if torch.sum(masks) == 0: continue 

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Post-process (Sigmoid + Threshold)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Convert to CPU/Numpy for plotting
            img_np = images.cpu().numpy()[0, 0] # Flair mod
            gt_np = masks.cpu().numpy()[0].sum(axis=0) # Merge channels for visualization
            pred_np = preds.cpu().numpy()[0].sum(axis=0)

            # Plotting
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].imshow(img_np, cmap='bone')
            ax[0].set_title("Input (FLAIR MRI)")
            
            ax[1].imshow(gt_np, cmap='jet')
            ax[1].set_title("Doctor's Label (Ground Truth)")
            
            ax[2].imshow(pred_np, cmap='jet')
            ax[2].set_title("U-Net++ Segmentation")
            
            for a in ax: a.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"research_gallery/paper_fig_{count}.png", dpi=300)
            plt.close()
            
            count += 1
            print(f"   - Saved Figure {count}/10")

    print(f"🎉 Success! 10 figures are ready in the 'research_gallery/' folder.")

if __name__ == "__main__":
    generate_paper_figures()
