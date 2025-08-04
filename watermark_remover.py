#!/usr/bin/env python3
"""
Watermark Removal Tool
Supports multiple techniques for removing watermarks from images.
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import restoration, morphology, filters
from skimage.segmentation import flood_fill
import warnings
warnings.filterwarnings('ignore')

class WatermarkRemover:
    def __init__(self, image_path: str):
        """Initialize with image path."""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.original = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
    def detect_watermark_mask(self, threshold: float = 0.8) -> np.ndarray:
        """
        Detect watermark areas using edge detection and pattern analysis.
        Returns a binary mask where watermarks are likely located.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast to make watermarks more visible
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Morphological operations to connect text elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours and filter by area and aspect ratio
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Typical text characteristics
                if 0.1 < aspect_ratio < 10 and area > 200:
                    cv2.fillPoly(mask, [contour], 255)
        
        # Apply morphological closing to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def detect_repeating_pattern(self, block_size: int = 64) -> np.ndarray:
        """
        Detect repeating watermark patterns using template matching.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Sample multiple small regions as potential watermark templates
        templates = []
        step = block_size // 2
        
        for y in range(0, self.height - block_size, step):
            for x in range(0, self.width - block_size, step):
                template = gray[y:y+block_size, x:x+block_size]
                
                # Check if template has enough variation (likely contains text)
                if np.std(template) > 20:
                    templates.append((template, x, y))
        
        # For each template, find matches across the image
        for template, orig_x, orig_y in templates:
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.7)
            
            # If we find multiple matches, it's likely a repeating watermark
            if len(locations[0]) > 3:
                for pt_y, pt_x in zip(locations[0], locations[1]):
                    cv2.rectangle(mask, (pt_x, pt_y), 
                                (pt_x + block_size, pt_y + block_size), 255, -1)
        
        return mask
    
    def inpaint_removal(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove watermarks using OpenCV's inpainting algorithms.
        """
        if mask is None:
            mask = self.detect_watermark_mask()
        
        # Try both inpainting algorithms
        result1 = cv2.inpaint(self.image, mask, 3, cv2.INPAINT_TELEA)
        result2 = cv2.inpaint(self.image, mask, 3, cv2.INPAINT_NS)
        
        # Use the result with better quality (less artifacts)
        # Simple metric: prefer the one with smoother gradients
        gray1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
        
        gradient1 = np.mean(cv2.Laplacian(gray1, cv2.CV_64F))
        gradient2 = np.mean(cv2.Laplacian(gray2, cv2.CV_64F))
        
        return result1 if abs(gradient1) < abs(gradient2) else result2
    
    def frequency_domain_removal(self) -> np.ndarray:
        """
        Remove watermarks using frequency domain filtering.
        """
        result = self.image.copy()
        
        for channel in range(3):
            # Apply FFT
            f_transform = np.fft.fft2(result[:,:,channel])
            f_shift = np.fft.fftshift(f_transform)
            
            # Create a mask to remove high-frequency noise (potential watermarks)
            crow, ccol = self.height // 2, self.width // 2
            mask = np.ones((self.height, self.width), np.uint8)
            
            # Remove high frequency components that might be watermarks
            mask[crow-30:crow+30, ccol-30:ccol+30] = 0
            
            # Apply mask and inverse FFT
            f_shift_masked = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_masked)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)
            
            # Normalize
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            result[:,:,channel] = img_back
        
        return result
    
    def adaptive_removal(self) -> np.ndarray:
        """
        Combine multiple techniques for better results.
        """
        # First, detect watermarks using multiple methods
        mask1 = self.detect_watermark_mask()
        mask2 = self.detect_repeating_pattern()
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply inpainting with the combined mask
        result = self.inpaint_removal(combined_mask)
        
        # Post-process with gentle filtering
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result, combined_mask
    
    def process_image(self, method: str = "adaptive") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process the image using the specified method.
        
        Args:
            method: "inpaint", "frequency", "adaptive"
        
        Returns:
            Tuple of (processed_image, mask_used)
        """
        if method == "inpaint":
            mask = self.detect_watermark_mask()
            return self.inpaint_removal(mask), mask
        elif method == "frequency":
            return self.frequency_domain_removal(), None
        elif method == "adaptive":
            return self.adaptive_removal()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save_result(self, result: np.ndarray, output_path: str):
        """Save the processed image."""
        cv2.imwrite(output_path, result)
        print(f"Result saved to: {output_path}")
    
    def show_comparison(self, result: np.ndarray, mask: Optional[np.ndarray] = None):
        """Display before/after comparison."""
        fig, axes = plt.subplots(1, 3 if mask is not None else 2, figsize=(15, 5))
        
        # Original
        axes[0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Result
        axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Watermark Removed')
        axes[1].axis('off')
        
        # Mask (if available)
        if mask is not None:
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('Detected Watermark Areas')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Remove watermarks from images')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path')
    parser.add_argument('-m', '--method', choices=['inpaint', 'frequency', 'adaptive'], 
                       default='adaptive', help='Removal method')
    parser.add_argument('--show', action='store_true', help='Show before/after comparison')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_no_watermark{ext}"
    
    try:
        # Initialize remover
        remover = WatermarkRemover(args.input)
        print(f"Processing image: {args.input}")
        print(f"Image size: {remover.width}x{remover.height}")
        print(f"Using method: {args.method}")
        
        # Process image
        result, mask = remover.process_image(args.method)
        
        # Save result
        remover.save_result(result, args.output)
        
        # Show comparison if requested
        if args.show:
            remover.show_comparison(result, mask)
            
        print("Watermark removal completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())