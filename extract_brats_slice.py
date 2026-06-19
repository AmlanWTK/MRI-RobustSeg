import numpy as np
import nibabel as nib
import cv2
import os
import argparse

def normalize_image(img):
    """Normalize MRI image to 0-255 uint8 format."""
    # Ignore zero background for normalization
    mask = img > 0
    if not np.any(mask):
        return np.zeros_like(img, dtype=np.uint8)
        
    img_min = np.min(img[mask])
    img_max = np.max(img[mask])
    
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)
        
    normalized = (img - img_min) / (img_max - img_min) * 255.0
    normalized[~mask] = 0
    return normalized.astype(np.uint8)

def extract_slice(brats_dir, slice_idx=75):
    """Extract a specific slice from a BraTS patient directory."""
    print(f"Loading BraTS data from: {brats_dir}")
    patient_id = os.path.basename(os.path.normpath(brats_dir))
    
    os.makedirs("test_images", exist_ok=True)
    
    modalities = {
        'FLAIR': f"{patient_id}_flair.nii",
        'T1': f"{patient_id}_t1.nii",
        'T1CE': f"{patient_id}_t1ce.nii",
        'T2': f"{patient_id}_t2.nii"
    }
    
    success = True
    for mod_name, filename in modalities.items():
        # Try both .nii and .nii.gz
        file_path = os.path.join(brats_dir, filename)
        if not os.path.exists(file_path):
            file_path += ".gz"
            if not os.path.exists(file_path):
                print(f"❌ Could not find {filename} or {filename}.gz in {brats_dir}")
                success = False
                continue
                
        print(f"Processing {mod_name}...")
        img_data = nib.load(file_path).get_fdata()
        
        if slice_idx >= img_data.shape[2]:
            print(f"Slice {slice_idx} is out of bounds (max {img_data.shape[2]-1})")
            success = False
            break
            
        # Extract the slice (transposed to look correct orientation-wise)
        slice_data = img_data[:, :, slice_idx]
        
        # BraTS images are usually 240x240, but let's rotate them to look like standard axial slices
        slice_data = np.rot90(slice_data)
        
        # Normalize to 8-bit image
        img_8bit = normalize_image(slice_data)
        
        # Save as PNG
        out_path = os.path.join("test_images", f"{mod_name}.png")
        cv2.imwrite(out_path, img_8bit)
        print(f"✅ Saved {out_path}")
        
    if success:
        print(f"\n🎉 Successfully extracted slice {slice_idx} into the 'test_images' folder!")
        print("You can now run 'python test_api.py' to test the model with this real patient data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PNG slices from BraTS NIfTI files")
    parser.add_argument("path", help="Path to the BraTS patient folder (e.g. BraTS20_Validation_001)")
    parser.add_argument("--slice", type=int, default=75, help="Slice number to extract (default: 75, middle of brain)")
    
    args = parser.parse_args()
    extract_slice(args.path, args.slice)
