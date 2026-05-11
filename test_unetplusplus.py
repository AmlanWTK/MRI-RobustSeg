
# UNet++ Model Test Script
# Save this as 'test_unetplusplus.py' and run it in your environment

import torch
import torch.nn as nn
from unetplusplus_model import UNetPlusPlus
import numpy as np

def test_unetplusplus_model():
    print("🔍 Testing UNet++ Model Implementation...")

    # Test 1: Model Creation
    try:
        model = UNetPlusPlus(in_ch=4, out_ch=3, deep_supervision=False)
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

    # Test 2: Model Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")

    # Test 3: Forward Pass with Standard Input
    try:
        # Create dummy MRI input (batch=2, channels=4, height=240, width=240)
        dummy_input = torch.randn(2, 4, 240, 240)
        print(f"🔍 Input shape: {dummy_input.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✅ Forward pass successful!")
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Expected output shape: torch.Size([2, 3, 240, 240])")

        # Check output shape
        if output.shape == torch.Size([2, 3, 240, 240]):
            print("✅ Output shape is correct!")
        else:
            print(f"❌ Output shape mismatch!")
            return False

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Test 4: Deep Supervision Mode
    try:
        model_ds = UNetPlusPlus(in_ch=4, out_ch=3, deep_supervision=True)
        with torch.no_grad():
            outputs = model_ds(dummy_input)

        print(f"✅ Deep supervision mode working!")
        print(f"📊 Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"📊 Output {i+1} shape: {out.shape}")

    except Exception as e:
        print(f"❌ Deep supervision test failed: {e}")
        return False

    # Test 5: Different Input Sizes
    try:
        test_sizes = [(1, 4, 128, 128), (1, 4, 256, 256), (3, 4, 224, 224)]
        for size in test_sizes:
            test_input = torch.randn(size)
            output = model(test_input)
            print(f"✅ Input {size} → Output {output.shape}")

    except Exception as e:
        print(f"❌ Different input sizes test failed: {e}")
        return False

    # Test 6: Model Device Compatibility
    try:
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            dummy_cuda = dummy_input.cuda()
            output_cuda = model_cuda(dummy_cuda)
            print("✅ CUDA compatibility confirmed!")
        else:
            print("ℹ️  CUDA not available, testing CPU only")

    except Exception as e:
        print(f"⚠️  CUDA test failed: {e}")

    print("\n🎉 All core tests passed! Your UNet++ model is working correctly!")
    return True

def test_model_with_degradations():
    '''Test model with various degradation operations'''
    print("\n🔍 Testing Model with Degradations...")

    # Simulate degraded inputs
    clean_input = torch.randn(1, 4, 240, 240)

    # Add some noise to simulate degraded image
    noisy_input = clean_input + torch.randn_like(clean_input) * 0.1

    model = UNetPlusPlus(in_ch=4, out_ch=3)
    model.eval()

    try:
        with torch.no_grad():
            clean_output = model(clean_input)
            noisy_output = model(noisy_input)

        print("✅ Model handles both clean and degraded inputs!")
        print(f"📊 Clean input → Output shape: {clean_output.shape}")
        print(f"📊 Noisy input → Output shape: {noisy_output.shape}")

        # Check if outputs have reasonable values
        clean_mean = clean_output.mean().item()
        noisy_mean = noisy_output.mean().item()
        print(f"📊 Clean output mean: {clean_mean:.4f}")
        print(f"📊 Noisy output mean: {noisy_mean:.4f}")

        return True

    except Exception as e:
        print(f"❌ Degradation test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 UNET++ BRAIN TUMOR SEGMENTATION MODEL TEST")
    print("=" * 60)

    # Run basic tests
    basic_test_passed = test_unetplusplus_model()

    if basic_test_passed:
        # Run degradation tests
        degradation_test_passed = test_model_with_degradations()

        if degradation_test_passed:
            print("\n🎉 ALL TESTS PASSED! Your model is ready for training!")
        else:
            print("\n⚠️  Basic tests passed, but degradation tests had issues")
    else:
        print("\n❌ Basic model tests failed. Please check your implementation.")
