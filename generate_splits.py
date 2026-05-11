import os
import glob
import pickle
import random
import argparse

"""
Dataset Split Generator
-----------------------
Scans a directory of preprocessed .npz files and creates 
train/val/test splits saved as .pkl files.
"""

def generate_splits(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    # Find all .npz files
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    random.shuffle(all_files)
    
    total = len(all_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    print(f"Total files: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    
    # Save splits
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, "train_files.pkl"), "wb") as f:
        pickle.dump(train_files, f)
    with open(os.path.join(output_dir, "val_files.pkl"), "wb") as f:
        pickle.dump(val_files, f)
    with open(os.path.join(output_dir, "test_files.pkl"), "wb") as f:
        pickle.dump(test_files, f)
        
    print(f"✅ Split files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset splits")
    parser.add_argument("--data", type=str, required=True, help="Directory containing preprocessed slices")
    parser.add_argument("--output", type=str, required=True, help="Directory to save split pickles")
    args = parser.parse_args()
    
    generate_splits(args.data, args.output)
