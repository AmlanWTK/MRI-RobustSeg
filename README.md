# Attention U-Net++ with Hybrid Degradation Augmentation for Brain Tumor Segmentation

**IEEE Conference Paper | NETCRYPT 2026**

---

## Abstract

This repository contains the official implementation of the paper:

> *"Robust Multimodal Brain Tumor Segmentation via Attention U-Net++ with Physics-Based Hybrid Degradation Augmentation"*

We propose a clinical-grade segmentation framework that combines the nested skip connections of U-Net++ with soft attention gates and a physics-informed Hybrid Degradation Augmentation (HDA) pipeline. Evaluated on the BraTS 2020 benchmark, the proposed framework achieves state-of-the-art performance with a Whole Tumor DSC of **0.8783** and a Tumor Core HD95 of **10.38 mm**, demonstrating the critical role of physics-based regularization in learning robust anatomical boundaries.

---

## Key Results (Table I)

| Configuration | DSC-WT ↑ | DSC-TC ↑ | DSC-ET ↑ | HD95-WT ↓ | HD95-TC ↓ | HD95-ET ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Standard U-Net (Baseline) | 0.8577 | 0.7632 | 0.7220 | 39.59 | 48.38 | 47.66 |
| U-Net++ (Nested) | 0.8688 | 0.7696 | 0.7272 | 33.75 | 33.40 | 37.81 |
| Att-UNet++ (w/o Augmentation) | 0.8681 | 0.7533 | 0.7115 | 38.85 | 50.30 | 50.91 |
| **Proposed Framework (Att-UNet++ + HDA)** | **0.8783** | **0.7898** | **0.7356** | **25.27** | **10.38** | **27.61** |

> DSC: Dice Similarity Coefficient (higher is better). HD95: 95th Percentile Hausdorff Distance in mm (lower is better).
> All metrics are computed at the 3D patient level on the BraTS 2020 validation set.

---

## Proposed Framework

### Architecture
- **Backbone**: Attention U-Net++ with 5-level encoder-decoder and nested dense skip connections.
- **Attention Gates**: Soft spatial attention at each decoder stage for selective feature recalibration.
- **Loss Function**: Combined Dice Loss + Focal Loss for class-imbalanced segmentation.

### Hybrid Degradation Augmentation (HDA)
The HDA pipeline simulates realistic MRI acquisition artifacts during training:
- **Rician Noise**: Correct noise model for MRI magnitude images.
- **Ghosting**: K-space echo artifact via periodic signal modulation.
- **Gibbs Ringing**: Truncation artifact via high-frequency K-space filtering.
- **Bias Field**: Low-frequency intensity inhomogeneity via polynomial field.

---

## Repository Structure

| File | Description |
|:---|:---|
| `train_segmentation.py` | **Primary training pipeline** with checkpoint resumption and metric logging |
| `train_segmentation_v2.py` | High-speed training pipeline using pre-sliced NumPy datasets |
| `ablation_study.py` | Full ablation study framework for all four model configurations |
| `final_evaluation.py` | 3D patient-level evaluation and paper table generation |
| `visualize_segmentation.py` | Qualitative segmentation visualization for paper figures |
| `models.py` | Baseline U-Net and U-Net++ architecture definitions |
| `augmentations.py` | Physics-based Hybrid Degradation Augmentation (HDA) module |
| `metrics.py` | DSC, IoU, and HD95 evaluation metric implementations |
| `dataset.py` | PyTorch Dataset for BraTS 2020 .npz files |
| `config.py` | Centralized hyperparameter and path configuration |

---

## Dataset

**BraTS 2020** (Brain Tumor Segmentation Challenge)
- **Source**: [Kaggle — awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- **Modalities**: FLAIR, T1, T1-contrast (T1ce), T2
- **Subjects**: 369 training patients
- **Labels**: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)

---

## Dependencies

```bash
pip install torch torchvision nibabel tqdm scipy numpy matplotlib pandas
```

---

## Usage

### Training (Kaggle)
1. Add the BraTS 2020 dataset and your checkpoint as Kaggle input datasets.
2. Set `CHECKPOINT_PATH` in `train_segmentation.py` to the correct path.
3. Run `train_segmentation.py` to begin training (or resume from checkpoint).

### Ablation Study
```bash
python ablation_study.py
```

### 3D Evaluation & Paper Table
```bash
python final_evaluation.py
```

### Visualization
```bash
python visualize_segmentation.py
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sarkar2026netcrypt,
  title     = {Robust Multimodal Brain Tumor Segmentation via Attention U-Net++ with Physics-Based Hybrid Degradation Augmentation},
  author    = {Amlan Sarkar},
  booktitle = {Proceedings of NETCRYPT 2026},
  year      = {2026},
  organization = {Khulna University of Engineering & Technology (KUET)}
}
```

---

## License

This project is released for academic research use only.
