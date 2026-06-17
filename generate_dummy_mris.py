import numpy as np
import cv2
import os

# Create a directory for test images
os.makedirs('test_images', exist_ok=True)

# Function to generate a synthetic MRI slice (grayscale image with a fake brain shape)
def generate_synthetic_mri(modality_name, intensity_factor):
    # Create a black background
    img = np.zeros((240, 240), dtype=np.uint8)
    
    # Draw a basic skull/brain outline (ellipse)
    cv2.ellipse(img, (120, 120), (90, 110), 0, 0, 360, int(150 * intensity_factor), -1)
    
    # Draw some fake brain structures
    cv2.ellipse(img, (100, 100), (30, 40), 45, 0, 360, int(100 * intensity_factor), -1)
    cv2.ellipse(img, (140, 140), (40, 30), -45, 0, 360, int(120 * intensity_factor), -1)
    
    # Add a fake tumor region
    cv2.circle(img, (120, 120), 25, int(200 * intensity_factor), -1)
    cv2.circle(img, (120, 120), 10, int(250 * intensity_factor), -1)
    
    # Save the image
    cv2.imwrite(f'test_images/{modality_name}.png', img)
    print(f"Generated test_images/{modality_name}.png")

# Generate the 4 required modalities with slight variations
generate_synthetic_mri('FLAIR', 0.8)
generate_synthetic_mri('T1', 0.6)
generate_synthetic_mri('T1CE', 1.0)
generate_synthetic_mri('T2', 0.9)

print("\n✅ Done! You can now find 4 sample images in the 'test_images' folder to upload in the app.")
