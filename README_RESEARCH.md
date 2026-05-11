# Brain Tumor Segmentation Research Project

## Project Overview
This project focuses on automated brain tumor segmentation using the **BraTS 2020** dataset. The primary architecture is **U-Net++**, evaluated for its robustness against **Hybrid Data Degradations**.

### **Data Preparation**
1. **Download**: The project now supports the [BraTS 2020 Kaggle Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).
2. **Install Dependencies**: `pip install nibabel tqdm`
3. **Preprocess**: Convert raw NIfTI to 2D slices:
   ```bash
   python preprocess_brats.py --input /path/to/raw/data --output ./preprocessed
   ```
4. **Split**: Generate training splits:
   ```bash
   python generate_splits.py --data ./preprocessed --output ./config
   ```
5. **Configure**: Update `DATA_ROOT` in `config.py`.

## Directory Structure
- `models.py`: Contains the U-Net++ architecture and reusable convolutional blocks.
- `metrics.py`: Standardized research metrics (Dice, IoU, HD95).
- `dataset.py`: PyTorch Dataset implementation for loading and augmenting .npz medical imaging data.
- `BrainTumorSegmentation.ipynb`: The main research driver for training, visualization, and results analysis.

## Core Metrics
We utilize three key metrics for evaluation:
1. **Dice Similarity Coefficient (DSC)**: Measures volume overlap.
2. **Hausdorff Distance (95th percentile)**: Measures boundary alignment (critical for clinical accuracy).
3. **Intersection over Union (IoU)**: Complementary overlap metric.

## Getting Started
1. Install dependencies: `pip install torch torchvision scipy numpy matplotlib h5py`
2. Ensure BraTS 2020 data is preprocessed into `.npz` format.
3. Open `BrainTumorSegmentation.ipynb` to run experiments.
