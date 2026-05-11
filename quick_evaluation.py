
# Quick UNet++ Evaluation Script (No matplotlib hanging)
# Save this as 'quick_evaluation.py'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from unetplusplus_model import UNetPlusPlus

class QuickEvaluator:
    def __init__(self, model_path=None):
        self.model = UNetPlusPlus(in_ch=4, out_ch=3, deep_supervision=False)
        self.model.eval()

        # Try to load weights with better error handling
        self.is_trained = False
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')

                # Try partial loading for mismatched architectures
                model_dict = self.model.state_dict()
                compatible_dict = {}

                for k, v in checkpoint.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            compatible_dict[k] = v
                        else:
                            print(f"⚠️ Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")
                    else:
                        print(f"⚠️ Key not found: {k}")

                if compatible_dict:
                    model_dict.update(compatible_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"✅ Loaded {len(compatible_dict)}/{len(checkpoint)} compatible layers")
                    self.is_trained = True
                else:
                    print("❌ No compatible layers found, using random weights")

            except Exception as e:
                print(f"❌ Error loading model: {e}")

        print(f"📊 Model Status: {'Partially Trained' if self.is_trained else 'Random Weights'}")

    def dice_coefficient(self, pred, target, smooth=1.0):
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > 0.5).float()

        intersection = (pred_binary * target).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean(dim=0)

    def evaluate_degradations(self, input_data, target_mask):
        print("🔍 Quick Degradation Evaluation...")
        print("-" * 60)

        degradations = {
            'Clean': lambda x: x,
            'Noisy': lambda x: x + torch.randn_like(x) * 0.1,
            'Blurred': lambda x: F.avg_pool2d(F.pad(x, (2,2,2,2), mode='replicate'), 5, stride=1),
            'Low_Res': lambda x: F.interpolate(F.interpolate(x, scale_factor=0.5), scale_factor=2),
            'Low_Contrast': lambda x: x * 0.6 + 0.2,
        }

        results = {}

        for name, degrade_fn in degradations.items():
            degraded = degrade_fn(input_data.clone())

            with torch.no_grad():
                pred = self.model(degraded)
                dice_scores = self.dice_coefficient(pred, target_mask)

            results[name] = {
                'dice_enhancing': dice_scores[0].item(),
                'dice_core': dice_scores[1].item(),
                'dice_whole': dice_scores[2].item(),
                'mean_dice': dice_scores.mean().item()
            }

            print(f"{name:12} | Dice: [{dice_scores[0]:.3f}, {dice_scores[1]:.3f}, {dice_scores[2]:.3f}] | Avg: {dice_scores.mean():.3f}")

        return results

    def test_model_basics(self):
        print("🔧 Testing Model Basics...")

        # Test forward pass
        dummy_input = torch.randn(2, 4, 240, 240)

        try:
            with torch.no_grad():
                output = self.model(dummy_input)
            print(f"✅ Forward pass successful: {dummy_input.shape} → {output.shape}")

            # Test probabilities
            probs = torch.sigmoid(output)
            print(f"📊 Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"📊 Probability range: [{probs.min():.4f}, {probs.max():.4f}]")

            return True
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False

def create_dummy_brain_data():
    '''Create realistic dummy brain MRI data'''
    # Create brain-like structure
    mri = torch.randn(1, 4, 240, 240) * 0.3 + 0.5

    # Add brain boundary
    center = 120
    radius = 100
    y, x = torch.meshgrid(torch.arange(240), torch.arange(240), indexing='ij')
    brain_mask = ((y - center)**2 + (x - center)**2) < radius**2

    # Apply brain mask to all modalities
    for i in range(4):
        mri[0, i] = mri[0, i] * brain_mask.float()

    # Create tumor masks
    tumor_mask = torch.zeros(1, 3, 240, 240)

    # Whole tumor (largest)
    whole_tumor = ((y - center)**2 + (x - center)**2) < 25**2
    tumor_mask[0, 2] = whole_tumor.float()

    # Core tumor  
    core_tumor = ((y - center)**2 + (x - center)**2) < 15**2
    tumor_mask[0, 1] = core_tumor.float()

    # Enhancing tumor
    enhancing_tumor = ((y - center)**2 + (x - center)**2) < 8**2
    tumor_mask[0, 0] = enhancing_tumor.float()

    return mri, tumor_mask

def main():
    print("=" * 60)
    print("🧠 QUICK UNET++ EVALUATION")
    print("=" * 60)

    # Initialize evaluator
    evaluator = QuickEvaluator('unetplusplus_brain_tumor.pth')

    # Test basic functionality
    if not evaluator.test_model_basics():
        print("❌ Basic tests failed, stopping evaluation")
        return

    # Create test data
    print("\n📊 Creating test data...")
    mri_data, tumor_masks = create_dummy_brain_data()
    print(f"✅ Created MRI: {mri_data.shape}, Masks: {tumor_masks.shape}")

    # Run degradation evaluation
    print("\n" + "=" * 60)
    results = evaluator.evaluate_degradations(mri_data, tumor_masks)

    # Summary
    print("\n📈 SUMMARY:")
    print("-" * 30)
    avg_performance = np.mean([r['mean_dice'] for r in results.values()])
    print(f"Average Dice across all degradations: {avg_performance:.3f}")

    # Check if model is actually learning something useful
    if avg_performance > 0.1:
        print("✅ Model shows reasonable performance!")
    elif evaluator.is_trained:
        print("⚠️ Model loaded but performance is low - may need retraining")
    else:
        print("ℹ️ Random weights - train the model for better performance")

    print("\n🎉 Quick evaluation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
