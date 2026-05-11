# Research Change Log

## [2026-05-07] - [x] Initial codebase modularization.
- [x] Transition support for BraTS 2020 Kaggle Dataset (NIfTI format).
- [x] Added `preprocess_brats.py` for NIfTI to NPZ slice conversion.
- [x] Added `generate_splits.py` for automated data splitting.

### **Logs**
#### **2026-05-08: Baseline Completion**
- [x] **Training**: Completed 5-epoch baseline training on clean BraTS data.
- [x] **Weights**: Saved `unetplusplus_brain_tumor.pth` (31MB).
- [x] **Metrics Fix**: Implemented `multiclass_dice_coeff` to exclude background bias.
- [x] **Simulation**: Implemented `augmentations.py` for Hybrid Degradation artifacts.
- [x] **Evaluation**: Created `evaluate.py` for automated metrics and paper figures.

### **Next Steps**
1.  **Baseline Evaluation**: Run `evaluate.py` to get the *real* tumor dice and HD95 scores.
2.  **Robustness Training**: Update `config.py` to `USE_AUGMENTATION = True` and start the final 50-epoch experiment.
3.  **Paper Drafting**: Begin compiling the Results tables.

## [2026-05-07] - Initial Refactoring for Publication
### Changes Made:
1. **Modularization**: Extracted core logic from the monolithic Jupyter Notebook into standalone Python modules:
   - `models.py`: Moved `UNetPlusPlus` and `ConvBlock`.
   - `metrics.py`: Moved `dice_coeff` and added `hd95` (Hausdorff Distance) and `iou_score`.
   - `dataset.py`: Moved `BrainTumorTorchDataset` and split loading logic.
2. **Metric Enhancement**: Added **95th percentile Hausdorff Distance (HD95)** to the evaluation suite. This is a critical metric for medical imaging papers.
3. **Documentation**: Created `README_RESEARCH.md` to explain the new modular project structure.

### Rationale:
- **Reproducibility**: Modular scripts are easier to version control and share with reviewers.
- **Scientific Rigor**: Adding HD95 ensures the project meets international standards for medical image segmentation research.
- **Maintainability**: New models (like standard U-Net or Attention U-Net) can now be added to `models.py` without cluttering the main notebook.

### [2026-05-07] - Implementation of Module Imports (Planned)
- **Task**: Update notebook to use the new modules.
- **Status**: Pending (Direct .ipynb editing restricted; manual update instructions to be provided).
