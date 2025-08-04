#!/usr/bin/env python3
"""
Example usage of the WatermarkRemover class.
"""

from watermark_remover import WatermarkRemover
import cv2

def remove_watermark_example(input_image_path: str, output_image_path: str = None):
    """
    Example function showing how to use the WatermarkRemover class.
    """
    # Initialize the watermark remover
    remover = WatermarkRemover(input_image_path)
    
    # Generate output path if not provided
    if output_image_path is None:
        import os
        base, ext = os.path.splitext(input_image_path)
        output_image_path = f"{base}_cleaned{ext}"
    
    print(f"Processing: {input_image_path}")
    print(f"Image size: {remover.width} x {remover.height}")
    
    # Try different methods
    methods = ["adaptive", "inpaint", "frequency"]
    
    for method in methods:
        print(f"\nTrying method: {method}")
        try:
            # Process the image
            result, mask = remover.process_image(method)
            
            # Save with method suffix
            base, ext = os.path.splitext(output_image_path)
            method_output = f"{base}_{method}{ext}"
            remover.save_result(result, method_output)
            
            print(f"✓ Success! Saved: {method_output}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print(f"\nDone! Check the output files for best results.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <input_image_path> [output_path]")
        print("Example: python example_usage.py my_image.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_watermark_example(input_path, output_path)