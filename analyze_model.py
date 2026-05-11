
# Model Architecture Analysis Script
# Save this as 'analyze_model.py'

import torch
from unetplusplus_model import UNetPlusPlus

def analyze_model_mismatch():
    print("🔍 ANALYZING MODEL ARCHITECTURE MISMATCH")
    print("=" * 60)

    # Current model architecture
    current_model = UNetPlusPlus(in_ch=4, out_ch=3, deep_supervision=False)
    current_state = current_model.state_dict()

    print(f"📊 Current Model Parameters: {len(current_state)}")
    print("\n🏗️ Current Model Architecture:")
    print("- UNet++ with nested skip connections")
    print("- Base filters: 32")
    print("- No upsampling layers (uses F.interpolate)")
    print("- Final layer: 'final.weight', 'final.bias'")

    # Try to load saved model
    try:
        saved_model = torch.load('unetplusplus_brain_tumor.pth', map_location='cpu')
        print(f"\n📊 Saved Model Parameters: {len(saved_model)}")

        print("\n❌ Keys in SAVED model but NOT in CURRENT model:")
        saved_only = set(saved_model.keys()) - set(current_state.keys())
        for key in sorted(saved_only):
            print(f"  - {key}")

        print("\n❌ Keys in CURRENT model but NOT in SAVED model:")
        current_only = set(current_state.keys()) - set(saved_model.keys())
        for key in sorted(current_only):
            print(f"  - {key}")

        print("\n⚠️ Keys with SHAPE MISMATCHES:")
        shape_mismatches = []
        for key in current_state.keys():
            if key in saved_model:
                if current_state[key].shape != saved_model[key].shape:
                    shape_mismatches.append(key)
                    print(f"  - {key}: {saved_model[key].shape} vs {current_state[key].shape}")

        print("\n📈 ANALYSIS RESULTS:")
        print("-" * 30)
        print(f"✅ Compatible keys: {len(current_state) - len(current_only) - len(shape_mismatches)}")
        print(f"❌ Missing keys: {len(saved_only)}")
        print(f"❌ Extra keys: {len(current_only)}")
        print(f"⚠️ Shape mismatches: {len(shape_mismatches)}")

        # Determine the issue
        if 'up3_1.weight' in saved_only:
            print("\n🔍 DIAGNOSIS:")
            print("Your saved model appears to be a DIFFERENT UNet architecture!")
            print("- Has 'up' layers (upsampling layers)")
            print("- Has 'out' layer instead of 'final'")
            print("- This suggests a standard U-Net, not UNet++")

            return 'different_architecture'
        else:
            print("\n🔍 DIAGNOSIS: Shape mismatch due to filter size difference")
            return 'filter_mismatch'

    except FileNotFoundError:
        print("❌ Saved model file not found!")
        return 'no_file'
    except Exception as e:
        print(f"❌ Error loading saved model: {e}")
        return 'load_error'

def create_compatible_model():
    '''Create a model that matches your saved weights'''
    print("\n🔧 CREATING COMPATIBLE MODEL")
    print("-" * 40)

    # This is likely what your original trained model looked like
    compatible_code = '''
# Original Model Architecture (save as 'original_unet.py')
import torch
import torch.nn as nn
import torch.nn.functional as F

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
'''

    print("💡 SOLUTION:")
    print("Your saved model is likely a standard U-Net, not UNet++!")
    print("The 'up' layers suggest ConvTranspose2d upsampling instead of interpolation.")
    print("\nOptions:")
    print("1. Use the compatible model code above")
    print("2. Retrain with your current UNet++ architecture")
    print("3. Convert between architectures (advanced)")

    return compatible_code

if __name__ == "__main__":
    diagnosis = analyze_model_mismatch()

    if diagnosis == 'different_architecture':
        compatible_code = create_compatible_model()

        # Save the compatible model code
        with open('original_unet.py', 'w') as f:
            f.write(compatible_code)
        print("\n✅ Created 'original_unet.py' with compatible architecture")

    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATION: Use the original architecture to load weights")
    print("   OR retrain with current UNet++ architecture")
    print("=" * 60)
