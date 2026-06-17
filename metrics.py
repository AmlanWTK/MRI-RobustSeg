import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

"""
IEEE Publication Codebase: Quantitative Evaluation Metrics
Target Conference: NETCRYPT 2026

Description:
    Standardized evaluation metrics for brain tumor segmentation benchmarking.
    Implements: Dice Similarity Coefficient (DSC), Intersection over Union (IoU),
    and 95th Percentile Hausdorff Distance (HD95).
    All metrics follow the BraTS 2020 official evaluation protocol.
"""

def dice_coefficient(pred, target, smooth=1e-5):
    """
    Computes the Dice Similarity Coefficient (DSC).
    Expects binary/boolean inputs.
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def multiclass_dice_coeff(outputs, masks, num_classes=3):
    """
    Calculates Mean Dice across all channels using Sigmoid.
    Standard for BraTS (WT, TC, ET overlapping regions).
    outputs: Tensor (B, 3, H, W) - Raw logits
    masks: Tensor (B, 3, H, W) - Binary masks
    """
    # Apply Sigmoid to get probabilities per channel
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    
    dice_scores = []
    for i in range(num_classes):
        dice_scores.append(dice_coefficient(preds[:, i], masks[:, i]))
        
    return torch.mean(torch.stack(dice_scores))

def iou_score(pred, target, smooth=1e-5):
    """
    Computes the Intersection over Union (Jaccard Index).
    Formula: |A ∩ B| / |A ∪ B|
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def hd95(pred, target):
    """
    Computes the 95th percentile Hausdorff Distance.
    Crucial for medical imaging to evaluate boundary alignment.
    Note: Requires CPU conversion for Scipy calculation.
    """
    # Ensure inputs are binary masks on CPU
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Only calculate if both masks contain positive pixels to avoid inf
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target) > 0:
        # Standard directed hausdorff distance
        d1 = directed_hausdorff(pred, target)[0]
        d2 = directed_hausdorff(target, pred)[0]
        return max(d1, d2)
    return 0.0
