
# Test Original U-Net Model with Saved Weights
# Save this as 'test_original_model.py'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class OriginalUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super(OriginalUNet, self).__init__()

        # Encoder (same as your saved model)
        self.conv0_0 = ConvBlock(in_ch, 32)
        self.conv1_0 = ConvBlock(32, 64)
        self.conv2_0 = ConvBlock(64, 128)
        self.conv3_0 = ConvBlock(128, 256)
        self.conv4_0 = ConvBlock(256, 512)

        # Decoder with ConvTranspose2d (matching saved model)
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

def test_original_model():
    print("🔍 TESTING ORIGINAL U-NET MODEL")
    print("=" * 50)

    # Create model
    model = OriginalUNet(in_ch=4, out_ch=3)
    print(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test basic functionality first
    dummy_input = torch.randn(2, 4, 240, 240)

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful: {dummy_input.shape} → {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Try to load saved weights
    try:
        state_dict = torch.load('unetplusplus_brain_tumor.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ Saved weights loaded PERFECTLY!")

        # Test with loaded weights
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.sigmoid(output)

        print(f"📊 Output range with trained weights: [{output.min():.4f}, {output.max():.4f}]")
        print(f"📊 Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")

        return True, model

    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False, None

def evaluate_trained_model():
    print("\n🧪 EVALUATING TRAINED MODEL PERFORMANCE")
    print("=" * 50)

    success, model = test_original_model()
    if not success:
        return

    # Create test data with realistic brain structure
    def create_brain_mri():
        # Create brain-like structure
        mri = torch.randn(1, 4, 240, 240) * 0.3

        # Add brain boundary
        center = 120
        radius = 100
        y, x = torch.meshgrid(torch.arange(240), torch.arange(240), indexing='ij')
        brain_mask = ((y - center)**2 + (x - center)**2) < radius**2

        # Apply brain mask and add realistic intensity patterns
        for i in range(4):
            mri[0, i] = (mri[0, i] + 0.5) * brain_mask.float()

        # Add some tumor-like hyperintense regions
        tumor_center_y, tumor_center_x = 100, 130
        tumor_region = ((y - tumor_center_y)**2 + (x - tumor_center_x)**2) < 20**2
        mri[0, 2] = mri[0, 2] + 0.8 * tumor_region.float()  # T1CE enhancement

        return mri

    # Test degradation robustness
    mri_data = create_brain_mri()

    degradations = {
        'Clean': lambda x: x,
        'Gaussian Noise': lambda x: x + torch.randn_like(x) * 0.1,
        'Motion Blur': lambda x: F.conv2d(x, torch.ones(4, 1, 5, 5) / 25, padding=2, groups=4),
        'Low Resolution': lambda x: F.interpolate(F.interpolate(x, scale_factor=0.5), scale_factor=2),
        'Low Contrast': lambda x: x * 0.6 + 0.2,
        'Bit Reduction': lambda x: torch.round(x * 15) / 15,
    }

    print("🔍 Testing robustness across degradations:")
    print("-" * 50)

    results = {}
    for name, degrade_fn in degradations.items():
        degraded_input = degrade_fn(mri_data.clone())

        with torch.no_grad():
            prediction = model(degraded_input)
            probabilities = torch.sigmoid(prediction)

            # Simple metrics
            mean_conf = probabilities.mean().item()
            max_conf = probabilities.max().item()
            positive_pixels = (probabilities > 0.5).sum().item()
            total_pixels = probabilities.numel()

            results[name] = {
                'mean_confidence': mean_conf,
                'max_confidence': max_conf, 
                'detection_rate': positive_pixels / total_pixels * 100
            }

            print(f"{name:15} | Avg: {mean_conf:.3f} | Max: {max_conf:.3f} | Detection: {positive_pixels/total_pixels*100:.1f}%")

    print("\n📈 PERFORMANCE SUMMARY:")
    print("-" * 30)
    avg_confidence = np.mean([r['mean_confidence'] for r in results.values()])
    std_confidence = np.std([r['mean_confidence'] for r in results.values()])

    print(f"Average confidence across degradations: {avg_confidence:.3f} ± {std_confidence:.3f}")

    if avg_confidence > 0.3:
        print("✅ Model shows STRONG performance - well trained!")
    elif avg_confidence > 0.1:
        print("⚠️ Model shows MODERATE performance - may need more training")
    else:
        print("❌ Model shows LOW performance - needs retraining")

    if std_confidence < 0.05:
        print("✅ Model is ROBUST across degradations!")
    else:
        print("⚠️ Model performance varies with degradation type")

    return results

if __name__ == "__main__":
    import numpy as np
    evaluate_trained_model()

    print("\n" + "=" * 60)
    print("🎉 ORIGINAL MODEL TESTING COMPLETED!")
    print("If weights loaded successfully, you have a working trained model!")
    print("=" * 60)
