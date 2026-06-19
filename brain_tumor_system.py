
# Complete Brain Tumor Segmentation System
# Save this as 'brain_tumor_system.py'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class BrainTumorUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super(BrainTumorUNet, self).__init__()

        # Encoder
        self.conv0_0 = ConvBlock(in_ch, 32)
        self.conv1_0 = ConvBlock(32, 64)
        self.conv2_0 = ConvBlock(64, 128)
        self.conv3_0 = ConvBlock(128, 256)
        self.conv4_0 = ConvBlock(256, 512)

        # Decoder
        self.up3_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock(256 + 256, 256)

        self.up2_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  
        self.conv2_2 = ConvBlock(128 + 128, 128)

        self.up1_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1_3 = ConvBlock(64 + 64, 64)

        self.up0_4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv0_4 = ConvBlock(32 + 32, 32)

        self.out = nn.Conv2d(32, out_ch, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x4_0 = self.conv4_0(self.maxpool(x3_0))

        # Decoder
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2_2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up1_3(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up0_4(x1_3)], 1))

        return self.out(x0_4)

class BrainTumorSegmentationSystem:
    def __init__(self, model_path='unetplusplus_brain_tumor.pth'):
        from train_segmentation import AttentionUNetPlusPlus
        self.model = AttentionUNetPlusPlus(in_ch=4, out_ch=3)
        self.model.eval()

        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # If the file is a full checkpoint dict, extract just the model weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict)
            print("✅ Trained model loaded successfully!")
            print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            self.is_trained = True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_trained = False

        self.class_names = ['Enhancing Tumor', 'Tumor Core', 'Whole Tumor']
        self.colors = ['red', 'blue', 'green']

    def preprocess_mri(self, mri_data):
        if isinstance(mri_data, np.ndarray):
            mri_tensor = torch.from_numpy(mri_data).float()
        else:
            mri_tensor = mri_data.float()

        # Ensure correct shape: (4, H, W)
        if mri_tensor.dim() == 3 and mri_tensor.shape[-1] == 4:
            mri_tensor = mri_tensor.permute(2, 0, 1)

        # Add batch dimension
        if mri_tensor.dim() == 3:
            mri_tensor = mri_tensor.unsqueeze(0)

        # Normalize each channel (ignore zero background like training code)
        for i in range(4):
            channel = mri_tensor[:, i]
            mask = channel > 0
            if mask.any():
                mean_val = channel[mask].mean()
                std_val = channel[mask].std()
                channel[mask] = (channel[mask] - mean_val) / (std_val + 1e-8)
            mri_tensor[:, i] = channel

        return mri_tensor

    def predict(self, mri_data, threshold=0.5):
        processed_input = self.preprocess_mri(mri_data)

        with torch.no_grad():
            prediction = self.model(processed_input)
            probabilities = torch.sigmoid(prediction)

        # Remove batch dimension
        probabilities = probabilities.squeeze(0)  # (3, H, W)
        binary_masks = (probabilities > threshold).float()

        return {
            'probabilities': probabilities.cpu().numpy(),
            'binary_masks': binary_masks.cpu().numpy(),
            'class_names': self.class_names,
            'input_shape': mri_data.shape,
            'model_confidence': float(probabilities.mean().item())
        }

    def generate_report(self, mri_data, save_path='tumor_analysis_report.json'):
        print("🧠 Analyzing brain tumor segmentation...")

        # Get predictions
        results = self.predict(mri_data)
        probabilities = results['probabilities']
        binary_masks = results['binary_masks']

        # Calculate metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_trained': self.is_trained,
            'overall_confidence': float(probabilities.mean()),
            'input_shape': list(mri_data.shape),
            'classes': {}
        }

        for i, class_name in enumerate(self.class_names):
            prob_map = probabilities[i]
            binary_mask = binary_masks[i]

            pixels_detected = int(binary_mask.sum())
            total_pixels = binary_mask.size
            volume_percentage = (pixels_detected / total_pixels) * 100

            metrics['classes'][class_name] = {
                'mean_confidence': float(prob_map.mean()),
                'max_confidence': float(prob_map.max()),
                'pixels_detected': pixels_detected,
                'volume_percentage': volume_percentage
            }

        # Clinical assessment
        enhancing_conf = metrics['classes']['Enhancing Tumor']['mean_confidence']
        core_conf = metrics['classes']['Tumor Core']['mean_confidence']
        whole_conf = metrics['classes']['Whole Tumor']['mean_confidence']

        if enhancing_conf > 0.3 or core_conf > 0.3 or whole_conf > 0.3:
            assessment = "TUMOR DETECTED - Recommend clinical review"
        elif enhancing_conf > 0.1 or core_conf > 0.1 or whole_conf > 0.1:
            assessment = "POSSIBLE ABNORMALITY - Consider additional imaging"
        else:
            assessment = "NO SIGNIFICANT ABNORMALITY DETECTED"

        metrics['clinical_assessment'] = assessment

        # Save report
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Analysis complete! Report saved to {save_path}")
        print(f"🏥 Clinical Assessment: {assessment}")

        return metrics, results

def demo_system():
    print("=" * 60)
    print("🧠 BRAIN TUMOR SEGMENTATION SYSTEM DEMO")
    print("=" * 60)

    # Initialize system
    system = BrainTumorSegmentationSystem('unetplusplus_brain_tumor.pth')

    # Create demo MRI data
    demo_mri = np.random.randn(240, 240, 4) * 0.3 + 0.5

    # Add brain structure
    center = 120
    radius = 100
    y, x = np.meshgrid(np.arange(240), np.arange(240))
    brain_mask = ((y - center)**2 + (x - center)**2) < radius**2

    for i in range(4):
        demo_mri[:, :, i] = demo_mri[:, :, i] * brain_mask

    # Add tumor-like region
    tumor_y, tumor_x = 100, 130
    tumor_region = ((y - tumor_y)**2 + (x - tumor_x)**2) < 15**2
    demo_mri[:, :, 2] += 0.8 * tumor_region  # T1CE enhancement

    print(f"📊 Demo MRI shape: {demo_mri.shape}")

    # Run analysis
    metrics, results = system.generate_report(demo_mri, 'demo_report.json')

    print("\n📊 RESULTS SUMMARY:")
    print("-" * 30)
    for class_name, class_metrics in metrics['classes'].items():
        print(f"{class_name}: {class_metrics['mean_confidence']:.3f} confidence, {class_metrics['volume_percentage']:.1f}% of brain")

    return system, metrics, results

if __name__ == "__main__":
    demo_system()
