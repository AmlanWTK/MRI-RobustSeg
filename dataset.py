import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

"""
Data Loading & Preprocessing
----------------------------
Handles BraTS 2020 NPZ files and apply augmentations/degradations.
"""

class BrainTumorTorchDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor Segmentation.
    Expects .npz files containing 'image' and 'mask' keys.
    """
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        
        # image shape: (H, W, C) -> (C, H, W)
        # mask shape: (H, W, C_mask) -> (C_mask, H, W)
        image = torch.from_numpy(data['image']).float().permute(2, 0, 1)
        mask = torch.from_numpy(data['mask']).float().permute(2, 0, 1)

        # Apply hybrid augmentations if provided
        if self.transform:
            image, mask, applied_augs = self.transform(image, mask)
            return image, mask, applied_augs

        return image, mask, []

def load_splits(splits_dir):
    """Utility to load pre-saved train/val/test split pickles."""
    with open(os.path.join(splits_dir, "train_files.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(splits_dir, "val_files.pkl"), "rb") as f:
        val = pickle.load(f)
    with open(os.path.join(splits_dir, "test_files.pkl"), "rb") as f:
        test = pickle.load(f)
    return train, val, test
