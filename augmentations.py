import torch
import numpy as np
import random
from scipy.ndimage import gaussian_filter

"""
Q1-RESEARCH GRADE: MRI Hybrid Degradation Pipeline
--------------------------------------------------
This module implements physics-based MRI artifact simulations.
Used for state-of-the-art robustness evaluation in medical imaging research.
Reference: "Physics-informed data augmentation for MRI" (Standard Q1 Methodology).
"""

class Q1HybridDegradation:
    def __init__(self, p=0.5, ghosting_prob=0.3, gibbs_prob=0.3, rician_std=(0.01, 0.05)):
        self.p = p
        self.ghosting_prob = ghosting_prob
        self.gibbs_prob = gibbs_prob
        self.rician_std = rician_std

    def _apply_ghosting(self, img, num_ghosts=4, intensity=0.15):
        """Simulates K-space signal echoes (N/2 ghosts)."""
        f = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f)
        rows, cols = img.shape
        # Zero out periodic rows in K-space to create echoes
        step = max(1, rows // num_ghosts)
        f_shift[::step, :] *= (1 - intensity)
        img_back = np.fft.ifft2(np.fft.ifftshift(f_shift))
        return np.abs(img_back)

    def _apply_gibbs_ringing(self, img, filter_size=0.7):
        """Simulates truncation artifacts due to limited K-space sampling."""
        f = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a circular mask to truncate high frequencies
        mask = np.zeros((rows, cols))
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        r = int(min(crow, ccol) * filter_size)
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2
        f_shift[~mask_area] = 0
        
        img_back = np.fft.ifft2(np.fft.ifftshift(f_shift))
        return np.abs(img_back)

    def _apply_rician_noise(self, img, std):
        """Mathematically correct noise model for MRI magnitude images."""
        x = img + np.random.normal(0, std, img.shape)
        y = np.random.normal(0, std, img.shape)
        return np.sqrt(x**2 + y**2)

    def _apply_bias_field(self, img):
        """Simulates intensity inhomogeneity (bias field)."""
        rows, cols = img.shape
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(x, y)
        
        # Random polynomial-based bias field
        poly = (random.uniform(-0.2, 0.2) * X + 
                random.uniform(-0.2, 0.2) * Y + 
                random.uniform(-0.1, 0.1) * X*Y + 1.0)
        return img * poly

    def _apply_motion_blur(self, img, kernel_size=7):
        """Simulates spatial motion blur (directional)."""
        kernel = np.zeros((kernel_size, kernel_size))
        # Random directional kernel
        angle = random.randint(0, 180)
        mid = kernel_size // 2
        kernel[mid, :] = 1.0
        # Simple rotation of the horizontal line kernel
        from scipy.ndimage import rotate
        kernel = rotate(kernel, angle, reshape=False)
        kernel /= kernel.sum()
        from scipy.signal import convolve2d
        return convolve2d(img, kernel, mode='same')

    def _apply_downsampling(self, img, scale=0.5):
        """Simulates low-resolution acquisition."""
        from scipy.ndimage import zoom
        h, w = img.shape
        low_res = zoom(img, scale, order=1)
        # Scale back up to original size
        return zoom(low_res, 1/scale, order=1)[:h, :w]

    def __call__(self, image_tensor, mask_tensor=None):
        """
        Args:
            image_tensor: Tensor (C, H, W)
            mask_tensor: Optional Tensor (C, H, W)
        Returns:
            Augmented Image, Mask, and list of applied artifacts
        """
        if random.random() > self.p:
            return image_tensor, mask_tensor, []

        img_np = image_tensor.numpy()
        applied = []

        # Apply artifacts per channel
        for c in range(img_np.shape[0]):
            # 1. Rician Noise
            std = random.uniform(*self.rician_std)
            img_np[c] = self._apply_rician_noise(img_np[c], std)
            
            # 2. Advanced Artifacts (Randomly pick 1 or 2 per channel)
            dice = random.random()
            if dice < 0.2:
                img_np[c] = self._apply_ghosting(img_np[c])
                applied.append("ghosting")
            elif dice < 0.4:
                img_np[c] = self._apply_gibbs_ringing(img_np[c])
                applied.append("gibbs_ringing")
            elif dice < 0.6:
                img_np[c] = self._apply_bias_field(img_np[c])
                applied.append("bias_field")
            elif dice < 0.8:
                img_np[c] = self._apply_motion_blur(img_np[c])
                applied.append("motion_blur")
            else:
                img_np[c] = self._apply_downsampling(img_np[c])
                applied.append("downsampling")

        img_np = np.clip(img_np, 0, 1)
        return torch.from_numpy(img_np).float(), mask_tensor, applied
