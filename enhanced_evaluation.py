
# Enhanced UNet++ Evaluation and Visualization Script


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from unetplusplus_model import UNetPlusPlus

class ModelEvaluator:
    def __init__(self, model_path=None):
        self.model = UNetPlusPlus(in_ch=4, out_ch=3, deep_supervision=False)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print("✅ Model weights loaded successfully!")
                self.is_trained = True
            except Exception as e:
                print(f"⚠️ Could not load weights: Using random initialization")
                self.is_trained = False
        else:
            print("⚠️ No model weights found: Using random initialization")
            self.is_trained = False

    def dice_coefficient(self, pred, target, smooth=1.0):
        '''Calculate Dice coefficient'''
        pred = torch.sigmoid(pred)
        pred_binary = (pred > 0.5).float()

        intersection = (pred_binary * target).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean(dim=0)  # Mean across batch

    def iou_score(self, pred, target, smooth=1.0):
        '''Calculate IoU (Intersection over Union)'''
        pred = torch.sigmoid(pred)
        pred_binary = (pred > 0.5).float()

        intersection = (pred_binary * target).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou.mean(dim=0)  # Mean across batch

    def evaluate_on_degradations(self, clean_input, target_mask=None):
        '''Evaluate model performance on various degradations'''

        # Define degradation operations
        degradations = {
            'Clean': lambda x: x,
            'Gaussian Noise': lambda x: x + torch.randn_like(x) * 0.1,
            'Motion Blur': lambda x: F.conv2d(x, torch.ones(4, 1, 5, 5) / 25, padding=2, groups=4),
            'Low Resolution': lambda x: F.interpolate(F.interpolate(x, scale_factor=0.5), scale_factor=2),
            'Low Contrast': lambda x: x * 0.5 + 0.25,
            'Bit Reduction': lambda x: torch.round(x * 15) / 15,  # 4-bit simulation
            'Salt Pepper': self._add_salt_pepper_noise,
        }

        results = {}

        print("🔍 Evaluating model performance on degradations...")
        print("-" * 60)

        for name, degradation_fn in degradations.items():
            degraded_input = degradation_fn(clean_input)

            with torch.no_grad():
                prediction = self.model(degraded_input)
                probabilities = torch.sigmoid(prediction)

            # Calculate metrics if target is provided
            metrics = {}
            if target_mask is not None:
                dice_scores = self.dice_coefficient(prediction, target_mask)
                iou_scores = self.iou_score(prediction, target_mask)

                metrics = {
                    'dice_enhancing': dice_scores[0].item(),
                    'dice_core': dice_scores[1].item(), 
                    'dice_whole': dice_scores[2].item(),
                    'iou_enhancing': iou_scores[0].item(),
                    'iou_core': iou_scores[1].item(),
                    'iou_whole': iou_scores[2].item(),
                }

            # Basic statistics
            metrics.update({
                'mean_prob': probabilities.mean().item(),
                'max_prob': probabilities.max().item(),
                'min_prob': probabilities.min().item(),
                'std_prob': probabilities.std().item(),
            })

            results[name] = metrics

            # Print results
            if target_mask is not None:
                print(f"{name:15} | Dice: [{metrics['dice_enhancing']:.3f}, {metrics['dice_core']:.3f}, {metrics['dice_whole']:.3f}] | "
                      f"IoU: [{metrics['iou_enhancing']:.3f}, {metrics['iou_core']:.3f}, {metrics['iou_whole']:.3f}]")
            else:
                print(f"{name:15} | Prob: mean={metrics['mean_prob']:.3f}, max={metrics['max_prob']:.3f}, std={metrics['std_prob']:.3f}")

        return results

    def _add_salt_pepper_noise(self, img, noise_factor=0.05):
        '''Add salt and pepper noise'''
        noisy = img.clone()
        noise = torch.rand_like(img)

        # Salt noise (white pixels)
        salt = noise < noise_factor / 2
        noisy[salt] = 1.0

        # Pepper noise (black pixels)  
        pepper = noise > (1 - noise_factor / 2)
        noisy[pepper] = 0.0

        return noisy

    def visualize_predictions(self, input_img, save_path='prediction_visualization.png'):
        '''Visualize model predictions on different degradations'''

        degradations = {
            'Clean': lambda x: x,
            'Noisy': lambda x: x + torch.randn_like(x) * 0.1,
            'Blurred': lambda x: F.conv2d(x, torch.ones(4, 1, 5, 5) / 25, padding=2, groups=4),
            'Low Res': lambda x: F.interpolate(F.interpolate(x, scale_factor=0.5), scale_factor=2),
        }

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']

        for idx, (name, degradation_fn) in enumerate(degradations.items()):
            degraded_input = degradation_fn(input_img)

            # Show first modality (FLAIR) of degraded input
            axes[0, idx].imshow(degraded_input[0, 0].cpu().numpy(), cmap='gray')
            axes[0, idx].set_title(f'{name} Input (FLAIR)')
            axes[0, idx].axis('off')

            # Get prediction
            with torch.no_grad():
                prediction = self.model(degraded_input)
                prob_mask = torch.sigmoid(prediction)[0]

                # Combine all tumor classes for visualization
                combined_mask = torch.max(prob_mask, dim=0)[0]

            # Show prediction
            axes[1, idx].imshow(combined_mask.cpu().numpy(), cmap='Reds')
            axes[1, idx].set_title(f'{name} Prediction')
            axes[1, idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Visualization saved to {save_path}")

def create_dummy_data():
    '''Create dummy MRI data for testing'''
    # Simulate a brain MRI with some tumor-like regions
    dummy_mri = torch.randn(1, 4, 240, 240) * 0.5 + 0.5

    # Create simple dummy masks (3 classes)
    dummy_mask = torch.zeros(1, 3, 240, 240)

    # Simulate tumor regions
    center_y, center_x = 120, 120
    y, x = torch.meshgrid(torch.arange(240), torch.arange(240), indexing='ij')

    # Whole tumor (largest)
    whole_tumor = ((y - center_y)**2 + (x - center_x)**2) < 30**2
    dummy_mask[0, 2] = whole_tumor.float()

    # Core tumor (medium)
    core_tumor = ((y - center_y)**2 + (x - center_x)**2) < 20**2
    dummy_mask[0, 1] = core_tumor.float()

    # Enhancing tumor (smallest)
    enhancing_tumor = ((y - center_y)**2 + (x - center_x)**2) < 10**2
    dummy_mask[0, 0] = enhancing_tumor.float()

    return dummy_mri, dummy_mask

if __name__ == "__main__":
    import os

    print("="*60)
    print("🧠 ENHANCED UNET++ EVALUATION")
    print("="*60)

    # Initialize evaluator
    evaluator = ModelEvaluator('unetplusplus_brain_tumor.pth')

    # Create dummy data for testing
    dummy_input, dummy_target = create_dummy_data()

    # Run evaluation on degradations
    results = evaluator.evaluate_on_degradations(dummy_input, dummy_target)

    # Create visualizations
    evaluator.visualize_predictions(dummy_input)

    print("\n" + "="*60)
    print("🎉 Enhanced evaluation completed!")
    print("Check the generated visualization image.")
    print("="*60)
