import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import UNetPlusPlus
from dataset import BrainTumorTorchDataset, load_splits
from metrics import multiclass_dice_coeff, hd95
from config import Config
import os

"""
Research Evaluation & Visualization Script
------------------------------------------
1. Loads the trained U-Net++ model.
2. Calculates real-world metrics (Multiclass Dice, HD95).
3. Saves visualization of segmentation results for the paper.
"""

def evaluate():
    device = torch.device(Config.DEVICE)
    print(f"Evaluating on: {device}")

    # 1. Load Data
    _, _, test_files = load_splits(Config.SPLIT_DIR)
    # We'll just evaluate on a subset of the test set for visualization
    test_dataset = BrainTumorTorchDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Load Model
    model = UNetPlusPlus(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS).to(device)
    model.load_state_dict(torch.load(Config.SAVE_MODEL_PATH, map_location=device))
    model.eval()

    print(f"Model loaded from {Config.SAVE_MODEL_PATH}")

    avg_dice = 0
    avg_hd95 = 0
    samples_with_tumor = 0
    max_eval = 50 # Increase search range to find tumor slices

    if not os.path.exists("results"):
        os.makedirs("results")

    with torch.no_grad():
        for i, (image, mask, _) in enumerate(test_loader):
            if samples_with_tumor >= 20 or i >= max_eval: break

            # Detection logic: check if any channel has tumor
            if torch.sum(mask > 0) == 0:
                continue

            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            
            # Metrics
            dice = multiclass_dice_coeff(output, mask)
            avg_dice += dice.item()
            
            # For visualization, we can collapse channels into a 2D map
            # or just show the Whole Tumor (Channel 0)
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).float().cpu().numpy()[0]
            target = mask.cpu().numpy()[0]
            
            # Collapse for visualization: WT + TC + ET
            pred_vis = np.sum(pred, axis=0) 
            target_vis = np.sum(target, axis=0)

            # HD95 (Calculate on Whole Tumor channel 0)
            h_dist = hd95(torch.from_numpy(pred[0]), torch.from_numpy(target[0]))
            avg_hd95 += h_dist
            
            samples_with_tumor += 1

            # Visualization
            if samples_with_tumor <= 5: 
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.title("Original MRI (FLAIR)")
                plt.imshow(image.cpu().numpy()[0, 0, :, :], cmap='bone')
                
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth (Merged)")
                plt.imshow(target_vis, cmap='jet')
                
                plt.subplot(1, 3, 3)
                plt.title(f"Prediction (Dice: {dice.item():.4f})")
                plt.imshow(pred_vis, cmap='jet')
                
                plt.savefig(f"results/prediction_{i}.png")
                plt.close()

    print("-" * 30)
    print(f"Evaluation Results ({samples_with_tumor} Tumor Slices):")
    if samples_with_tumor > 0:
        print(f"Mean Tumor Dice: {avg_dice/samples_with_tumor:.4f}")
        print(f"Mean HD95: {avg_hd95/samples_with_tumor:.4f} pixels")
    else:
        print("No tumor slices found in the search range.")
    print("-" * 30)
    print(f"Visualizations saved to the 'results/' folder.")

if __name__ == "__main__":
    evaluate()
