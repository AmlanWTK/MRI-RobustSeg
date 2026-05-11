
# Real-world Inference Pipeline for Brain Tumor Segmentation
# Save this as 'inference_pipeline.py'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unetplusplus_model import UNetPlusPlus

class BrainTumorSegmentationPipeline:
    def __init__(self, model_path=None):
        '''Initialize the segmentation pipeline'''
        self.model = UNetPlusPlus(in_ch=4, out_ch=3, deep_supervision=False)
        self.model.eval()

        # Load weights if available
        if model_path:
            try:
                # Try different loading strategies for weight mismatch
                checkpoint = torch.load(model_path, map_location='cpu')

                # If direct loading fails, try partial loading
                try:
                    self.model.load_state_dict(checkpoint)
                    print("✅ Full model weights loaded successfully!")
                    self.is_trained = True
                except RuntimeError as e:
                    print("⚠️ Full loading failed, trying partial loading...")
                    # Load compatible layers only
                    model_dict = self.model.state_dict()
                    compatible_dict = {k: v for k, v in checkpoint.items() 
                                     if k in model_dict and v.shape == model_dict[k].shape}

                    model_dict.update(compatible_dict)
                    self.model.load_state_dict(model_dict)

                    print(f"✅ Loaded {len(compatible_dict)} compatible layers out of {len(checkpoint)}")
                    self.is_trained = len(compatible_dict) > 0

            except Exception as e:
                print(f"❌ Could not load weights: {e}")
                print("   Using random initialization...")
                self.is_trained = False
        else:
            self.is_trained = False

        print(f"📊 Model status: {'Trained' if self.is_trained else 'Random weights'}")

    def preprocess_mri(self, mri_data):
        '''Preprocess MRI data for model input'''
        # Expecting mri_data as numpy array with shape (H, W, 4) or (4, H, W)

        if isinstance(mri_data, np.ndarray):
            mri_tensor = torch.from_numpy(mri_data).float()
        else:
            mri_tensor = mri_data.float()

        # Ensure correct shape: (4, H, W)
        if mri_tensor.dim() == 3 and mri_tensor.shape[-1] == 4:
            mri_tensor = mri_tensor.permute(2, 0, 1)  # (H, W, 4) -> (4, H, W)

        # Add batch dimension
        if mri_tensor.dim() == 3:
            mri_tensor = mri_tensor.unsqueeze(0)  # (1, 4, H, W)

        # Normalize each channel
        for i in range(4):
            channel = mri_tensor[:, i]
            mean_val = channel.mean()
            std_val = channel.std()
            mri_tensor[:, i] = (channel - mean_val) / (std_val + 1e-8)

        return mri_tensor

    def postprocess_prediction(self, prediction, threshold=0.5):
        '''Convert model output to segmentation masks'''
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(prediction)

        # Create binary masks
        binary_masks = (probabilities > threshold).float()

        return probabilities, binary_masks

    def predict(self, mri_data, return_probabilities=True):
        '''Main prediction function'''
        # Preprocess input
        processed_input = self.preprocess_mri(mri_data)

        # Run inference
        with torch.no_grad():
            prediction = self.model(processed_input)

        # Postprocess output
        probabilities, binary_masks = self.postprocess_prediction(prediction)

        # Remove batch dimension
        probabilities = probabilities.squeeze(0)  # (3, H, W)
        binary_masks = binary_masks.squeeze(0)    # (3, H, W)

        if return_probabilities:
            return {
                'probabilities': probabilities.cpu().numpy(),
                'binary_masks': binary_masks.cpu().numpy(),
                'class_names': ['Enhancing Tumor', 'Tumor Core', 'Whole Tumor']
            }
        else:
            return binary_masks.cpu().numpy()

    def predict_with_degradation_handling(self, mri_data, degradation_type='auto'):
        '''Predict with automatic degradation detection and handling'''

        # Auto-detect degradation type based on image statistics
        if degradation_type == 'auto':
            degradation_type = self._detect_degradation(mri_data)

        # Apply preprocessing based on degradation type
        if degradation_type == 'noisy':
            # Apply light denoising
            mri_data = self._denoise_light(mri_data)
        elif degradation_type == 'low_contrast':
            # Apply contrast enhancement
            mri_data = self._enhance_contrast(mri_data)
        elif degradation_type == 'blurred':
            # Apply light sharpening
            mri_data = self._sharpen_light(mri_data)

        return self.predict(mri_data)

    def _detect_degradation(self, mri_data):
        '''Simple degradation detection'''
        if isinstance(mri_data, np.ndarray):
            data = torch.from_numpy(mri_data).float()
        else:
            data = mri_data

        # Simple heuristics
        variance = data.var().item()
        mean_intensity = data.mean().item()

        if variance < 0.1:
            return 'low_contrast'
        elif variance > 0.5:
            return 'noisy'
        else:
            return 'clean'

    def _denoise_light(self, data):
        '''Light denoising using Gaussian blur'''
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Light Gaussian smoothing
        kernel = torch.ones(1, 1, 3, 3) / 9
        if data.dim() == 3:
            data = data.unsqueeze(0)

        denoised = F.conv2d(data, kernel, padding=1)
        return denoised.squeeze(0) if denoised.size(0) == 1 else denoised

    def _enhance_contrast(self, data):
        '''Simple contrast enhancement'''
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Simple linear contrast stretch
        min_val = data.min()
        max_val = data.max()
        enhanced = (data - min_val) / (max_val - min_val + 1e-8)
        return enhanced * 1.2  # Slight enhancement

    def _sharpen_light(self, data):
        '''Light sharpening'''
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Simple unsharp mask
        blurred = self._denoise_light(data)
        sharpened = data + 0.3 * (data - blurred)
        return sharpened

    def calculate_tumor_volume(self, binary_masks, voxel_spacing=(1.0, 1.0, 1.0)):
        '''Calculate tumor volume in mm³'''
        volumes = {}
        class_names = ['Enhancing Tumor', 'Tumor Core', 'Whole Tumor']

        for i, class_name in enumerate(class_names):
            voxel_count = binary_masks[i].sum()
            volume_mm3 = voxel_count * np.prod(voxel_spacing)
            volumes[class_name] = volume_mm3

        return volumes

    def generate_report(self, mri_data, save_path=None):
        '''Generate a comprehensive segmentation report'''
        results = self.predict(mri_data)

        probabilities = results['probabilities']
        binary_masks = results['binary_masks']
        class_names = results['class_names']

        # Calculate basic statistics
        report = {
            'model_status': 'Trained' if self.is_trained else 'Random weights',
            'input_shape': mri_data.shape,
            'classes': {}
        }

        for i, class_name in enumerate(class_names):
            prob_map = probabilities[i]
            binary_mask = binary_masks[i]

            report['classes'][class_name] = {
                'mean_probability': float(prob_map.mean()),
                'max_probability': float(prob_map.max()),
                'pixels_detected': int(binary_mask.sum()),
                'percentage_of_brain': float(binary_mask.mean() * 100)
            }

        # Calculate volumes (assuming 1mm³ voxels)
        volumes = self.calculate_tumor_volume(binary_masks)
        for class_name, volume in volumes.items():
            report['classes'][class_name]['volume_mm3'] = float(volume)

        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Report saved to {save_path}")

        return report

# Example usage function
def example_usage():
    '''Example of how to use the pipeline'''
    print("🧠 Brain Tumor Segmentation Pipeline Example")
    print("-" * 50)

    # Initialize pipeline
    pipeline = BrainTumorSegmentationPipeline('unetplusplus_brain_tumor.pth')

    # Create dummy MRI data (in real use, load from .nii, .dcm, etc.)
    dummy_mri = np.random.randn(240, 240, 4).astype(np.float32)
    print(f"📊 Input MRI shape: {dummy_mri.shape}")

    # Run prediction
    results = pipeline.predict(dummy_mri)

    print("\n📊 Prediction Results:")
    for i, class_name in enumerate(results['class_names']):
        prob_mean = results['probabilities'][i].mean()
        pixels_detected = results['binary_masks'][i].sum()
        print(f"  {class_name}: {pixels_detected:.0f} pixels, avg prob: {prob_mean:.3f}")

    # Generate comprehensive report
    report = pipeline.generate_report(dummy_mri, 'segmentation_report.json')

    print("\n✅ Segmentation completed successfully!")
    return results, report

if __name__ == "__main__":
    example_usage()
