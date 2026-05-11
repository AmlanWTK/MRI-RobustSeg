import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import UNetPlusPlus
from dataset import BrainTumorTorchDataset
from metrics import dice_coefficient, iou_score
from config import Config
import os

def train():
    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Prepare Data
    from dataset import load_splits
    from augmentations import HybridDegradation
    train_files, val_files, test_files = load_splits(Config.SPLIT_DIR)
    
    transform = HybridDegradation() if Config.USE_AUGMENTATION else None
    
    print(f"Loaded {len(train_files)} training samples and {len(val_files)} validation samples.")

    train_dataset = BrainTumorTorchDataset(train_files, transform=transform)
    val_dataset = BrainTumorTorchDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 3. Initialize Model, Criterion, Optimizer
    model = UNetPlusPlus(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop
    from tqdm import tqdm
    from metrics import multiclass_dice_coeff
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        for i, (images, masks, _) in enumerate(train_pbar):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        val_loss, val_dice = validate(model, val_loader, device, criterion)
        
        print(f"\nSummary Epoch [{epoch+1}/{Config.NUM_EPOCHS}] - "
              f"Avg Loss: {epoch_loss/len(train_loader):.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Dice (Avg): {val_dice:.4f}\n")

        # Save Checkpoint every epoch to protect progress
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), "latest_model.pth")
        print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), Config.SAVE_MODEL_PATH)
    print(f"Training complete. Model saved to {Config.SAVE_MODEL_PATH}")

def validate(model, loader, device, criterion):
    model.eval()
    val_loss = 0
    val_dice = 0
    from tqdm import tqdm
    from metrics import multiclass_dice_coeff
    val_pbar = tqdm(loader, desc="Evaluation", leave=False)
    with torch.no_grad():
        for images, masks, _ in val_pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Use new multiclass metric
            val_dice += multiclass_dice_coeff(outputs, masks)

    return val_loss / len(loader), val_dice / len(loader)

if __name__ == "__main__":
    train()
