import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse

"""
BraTS 2020 NIfTI Preprocessing Script
-------------------------------------
Converts raw .nii.gz volumes into 2D .npz slices for training.
Each .npz file contains:
- image: (240, 240, 4) -> [FLAIR, T1, T1ce, T2]
- mask: (240, 240, 3) -> [WT, TC, ET]
"""

def preprocess_brats(raw_data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all patient folders
    patient_dirs = glob.glob(os.path.join(raw_data_dir, "BraTS20_Training_*"))
    print(f"Found {len(patient_dirs)} patients in {raw_data_dir}")

    for patient_dir in tqdm(patient_dirs, desc="Processing Patients"):
        p_id = os.path.basename(patient_dir)
        
        # Load modalities
        try:
            flair = nib.load(os.path.join(patient_dir, f"{p_id}_flair.nii")).get_fdata()
            t1 = nib.load(os.path.join(patient_dir, f"{p_id}_t1.nii")).get_fdata()
            t1ce = nib.load(os.path.join(patient_dir, f"{p_id}_t1ce.nii")).get_fdata()
            t2 = nib.load(os.path.join(patient_dir, f"{p_id}_t2.nii")).get_fdata()
            seg = nib.load(os.path.join(patient_dir, f"{p_id}_seg.nii")).get_fdata()
        except FileNotFoundError as e:
            print(f"Skipping {p_id} due to missing files: {e}")
            continue

        # Iterate through slices (Z-axis)
        # BraTS images are usually (240, 240, 155)
        for i in range(seg.shape[2]):
            mask_slice = seg[:, :, i]
            
            # Skip empty slices to save space and focus training on tumor regions
            if np.max(mask_slice) == 0:
                continue
                
            # Prepare image (4 channels)
            img_slice = np.stack([
                flair[:, :, i],
                t1[:, :, i],
                t1ce[:, :, i],
                t2[:, :, i]
            ], axis=-1)

            # Prepare masks (3 channels: WT, TC, ET)
            # Label 1: NCR/NET, Label 2: ED, Label 4: ET
            wt = np.where(mask_slice > 0, 1, 0)
            tc = np.where((mask_slice == 1) | (mask_slice == 4), 1, 0)
            et = np.where(mask_slice == 4, 1, 0)
            
            mask_multi = np.stack([wt, tc, et], axis=-1)

            # Save as .npz
            slice_name = f"{p_id}_slice_{i}.npz"
            np.savez_compressed(
                os.path.join(output_dir, slice_name),
                image=img_slice.astype(np.float32),
                mask=mask_multi.astype(np.uint8)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BraTS NIfTI to NPZ slices")
    parser.add_argument("--input", type=str, required=True, help="Path to raw BraTS data")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed slices")
    args = parser.parse_args()

    preprocess_brats(args.input, args.output)
