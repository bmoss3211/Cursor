#!/usr/bin/env python3
"""
Script to remove crosshatch/diamond line patterns from images.
This script attempts to detect and remove repetitive diagonal line patterns
commonly used as watermarks or overlays.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

def remove_crosshatch_lines(image_path, output_path=None):
    """
    Remove crosshatch/diamond line patterns from an image.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path for the output image (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return False
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {img.shape}")
        
        # Convert to different color spaces for better line detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for white/light colored lines
        # Adjust these thresholds based on the line color
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(white_mask, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        # Create a mask for the detected lines
        line_mask = np.zeros_like(gray)
        
        if lines is not None:
            print(f"Detected {len(lines)} line segments")
            
            # Draw detected lines on the mask
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle to filter diagonal lines
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Keep only diagonal lines (around 45° and -45°)
                if abs(abs(angle) - 45) < 20 or abs(abs(angle) - 135) < 20:
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)
        
        # Dilate the mask to ensure complete line coverage
        kernel = np.ones((3, 3), np.uint8)
        line_mask = cv2.dilate(line_mask, kernel, iterations=1)
        
        # Inpaint to remove the lines
        result = cv2.inpaint(img, line_mask, 3, cv2.INPAINT_TELEA)
        
        # Set output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_no_lines{input_path.suffix}"
        
        # Save the result
        cv2.imwrite(str(output_path), result)
        print(f"Processed image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def remove_crosshatch_advanced(image_path, output_path=None):
    """
    Advanced method to remove crosshatch patterns using morphological operations.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create kernels for diagonal line detection
        kernel_diag1 = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=np.uint8)
        
        kernel_diag2 = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0]], dtype=np.uint8)
        
        # Detect diagonal patterns
        morph1 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_diag1)
        morph2 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_diag2)
        
        # Combine the detected patterns
        combined = cv2.bitwise_or(morph1, morph2)
        
        # Threshold to create binary mask
        _, mask = cv2.threshold(combined, 200, 255, cv2.THRESH_BINARY)
        
        # Inpaint the detected areas
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # Set output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_advanced_no_lines{input_path.suffix}"
        
        # Save the result
        cv2.imwrite(str(output_path), result)
        print(f"Advanced processed image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error in advanced processing: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Remove crosshatch/diamond line patterns from images")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path (optional)")
    parser.add_argument("-a", "--advanced", action="store_true", 
                       help="Use advanced morphological method")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Process the image
    if args.advanced:
        success = remove_crosshatch_advanced(args.input, args.output)
    else:
        success = remove_crosshatch_lines(args.input, args.output)
    
    if success:
        print("Image processing completed successfully!")
    else:
        print("Image processing failed.")

if __name__ == "__main__":
    main()