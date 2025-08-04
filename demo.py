#!/usr/bin/env python3
"""
Demonstration of the watermark removal tool.
Shows how to process images with watermarks similar to the Six Flags image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from watermark_remover import WatermarkRemover
import os

def create_demo_watermarked_image():
    """
    Create a demo image with watermarks similar to the Six Flags image
    to demonstrate the removal capabilities.
    """
    # Create a sample image (simulating a photo background)
    height, width = 600, 800
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add some "scenery" - gradients and shapes to simulate a real photo
    y, x = np.ogrid[:height, :width]
    
    # Sky gradient
    sky = np.linspace(180, 120, height)
    img[:, :, 0] = sky[:, np.newaxis]  # Blue channel
    img[:, :, 1] = sky[:, np.newaxis] * 0.8  # Green channel
    img[:, :, 2] = sky[:, np.newaxis] * 0.6  # Red channel
    
    # Add some "ground"
    ground_start = int(height * 0.7)
    img[ground_start:, :, 0] = 60  # Darker for ground
    img[ground_start:, :, 1] = 80
    img[ground_start:, :, 2] = 40
    
    # Add watermarks similar to "DOWNLOAD & SHARE" pattern
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    thickness = 1
    
    # Create diamond pattern of watermarks
    text = "DOWNLOAD & SHARE"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Add watermarks in a diamond grid pattern
    for y in range(0, height + 100, 80):
        for x in range(0, width + 200, 120):
            # Offset every other row for diamond pattern
            offset_x = 60 if (y // 80) % 2 == 1 else 0
            pos_x = x + offset_x
            pos_y = y
            
            if 0 <= pos_x < width - text_size[0] and 0 <= pos_y < height:
                # Add semi-transparent watermark
                overlay = img.copy()
                cv2.putText(overlay, text, (pos_x, pos_y), font, font_scale, color, thickness)
                # Blend with original (making watermark semi-transparent)
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    return img

def demonstrate_watermark_removal():
    """
    Demonstrate the watermark removal process.
    """
    print("🎯 Watermark Removal Demonstration")
    print("=" * 50)
    
    # Create demo image
    print("📷 Creating demo watermarked image...")
    demo_img = create_demo_watermarked_image()
    demo_path = "demo_watermarked.jpg"
    cv2.imwrite(demo_path, demo_img)
    print(f"✓ Demo image saved as: {demo_path}")
    
    # Initialize watermark remover
    print("\n🔧 Initializing watermark remover...")
    remover = WatermarkRemover(demo_path)
    print(f"✓ Image loaded: {remover.width}x{remover.height} pixels")
    
    # Test different methods
    methods = [
        ("adaptive", "🧠 Adaptive (combines multiple techniques)"),
        ("inpaint", "🎨 Inpainting (fills watermarked areas)"),
        ("frequency", "📊 Frequency domain filtering")
    ]
    
    results = {}
    
    for method, description in methods:
        print(f"\n{description}")
        print("-" * 40)
        
        try:
            # Process image
            result, mask = remover.process_image(method)
            
            # Save result
            output_path = f"demo_result_{method}.jpg"
            remover.save_result(result, output_path)
            
            # Save mask if available
            if mask is not None:
                mask_path = f"demo_mask_{method}.jpg"
                cv2.imwrite(mask_path, mask)
                print(f"  🎯 Watermark mask saved: {mask_path}")
            
            results[method] = (result, mask)
            print(f"  ✅ Success! Result saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Create comparison visualization
    print("\n📊 Creating comparison visualization...")
    create_comparison_plot(demo_img, results)
    
    # Provide usage instructions
    print("\n" + "=" * 50)
    print("🚀 HOW TO USE WITH YOUR IMAGE:")
    print("=" * 50)
    print("1. Save your image (like the Six Flags photo) to this directory")
    print("2. Run one of these commands:")
    print("   • python3 watermark_remover.py your_image.jpg")
    print("   • python3 example_usage.py your_image.jpg")
    print("   • python3 watermark_remover.py your_image.jpg --show")
    print("\n3. The tool will:")
    print("   • Detect watermark areas automatically")
    print("   • Apply the best removal technique")
    print("   • Save the cleaned image")
    print("\n💡 Tips for best results:")
    print("   • Use 'adaptive' method for complex watermarks")
    print("   • Use 'inpaint' for text-based watermarks")
    print("   • Try 'frequency' for repeating patterns")
    
    return results

def create_comparison_plot(original, results):
    """
    Create a visual comparison of the original and processed images.
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Watermark Removal Comparison', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original (Watermarked)', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Results
        methods = ['adaptive', 'inpaint', 'frequency']
        for i, method in enumerate(methods):
            if method in results:
                result, mask = results[method]
                
                # Show result
                axes[0, i+1 if i < 2 else 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                axes[0, i+1 if i < 2 else 1].set_title(f'{method.capitalize()} Method', fontweight='bold')
                axes[0, i+1 if i < 2 else 1].axis('off')
                
                # Show mask if available
                if mask is not None:
                    row = 1 if i < 2 else 1
                    col = i if i < 2 else 2
                    axes[row, col].imshow(mask, cmap='gray')
                    axes[row, col].set_title(f'{method.capitalize()} Mask', fontweight='bold')
                    axes[row, col].axis('off')
        
        # Hide unused subplots
        if len(results) < 3:
            for i in range(len(results), 3):
                axes[0, i+1].axis('off')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Comparison saved as: comparison_results.png")
        
    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")

if __name__ == "__main__":
    demonstrate_watermark_removal()
    print("\n🎉 Demonstration complete!")
    print("📁 Check the generated files:")
    print("   • demo_watermarked.jpg (original with watermarks)")
    print("   • demo_result_*.jpg (cleaned versions)")
    print("   • demo_mask_*.jpg (detected watermark areas)")
    print("   • comparison_results.png (side-by-side comparison)")