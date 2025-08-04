# Watermark Removal Tool

A Python-based tool for removing watermarks from images using multiple computer vision techniques.

## Features

- **Multiple Removal Methods**: 
  - Adaptive (combines multiple techniques)
  - Inpainting (fills watermarked areas intelligently)
  - Frequency domain filtering (removes repetitive patterns)

- **Automatic Detection**: Automatically detects watermark areas using:
  - Edge detection and pattern analysis
  - Template matching for repeating patterns
  - Morphological operations for text recognition

- **Smart Processing**: Uses advanced algorithms to minimize artifacts and preserve image quality

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Basic usage (uses adaptive method by default)
python watermark_remover.py input_image.jpg

# Specify output file
python watermark_remover.py input_image.jpg -o cleaned_image.jpg

# Choose specific method
python watermark_remover.py input_image.jpg -m inpaint

# Show before/after comparison
python watermark_remover.py input_image.jpg --show
```

### Available Methods

- `adaptive` (default): Combines multiple detection and removal techniques
- `inpaint`: Uses OpenCV's inpainting algorithms (best for text watermarks)
- `frequency`: Uses FFT filtering (good for repetitive patterns)

### Python API

```python
from watermark_remover import WatermarkRemover

# Initialize with your image
remover = WatermarkRemover("your_image.jpg")

# Process using adaptive method
result, mask = remover.process_image("adaptive")

# Save the result
remover.save_result(result, "output.jpg")

# Show comparison
remover.show_comparison(result, mask)
```

### Example Usage Script

Run the example script to test all methods:

```bash
python example_usage.py your_image.jpg
```

This will create three output files using different methods so you can compare results.

## How It Works

### 1. Watermark Detection
- **Edge Detection**: Uses Canny edge detection to find text-like features
- **Pattern Matching**: Template matching to find repeating watermark patterns
- **Morphological Analysis**: Identifies text characteristics (aspect ratios, areas)

### 2. Removal Techniques
- **Inpainting**: Fills detected watermark areas using surrounding pixel information
- **Frequency Filtering**: Removes high-frequency components that typically contain watermarks
- **Adaptive Combination**: Merges multiple detection methods and applies optimal removal

### 3. Post-Processing
- Bilateral filtering to smooth results while preserving edges
- Quality assessment to choose best inpainting algorithm
- Morphological operations to clean up detection masks

## Tips for Best Results

1. **For text watermarks**: Use `inpaint` or `adaptive` methods
2. **For repeating patterns**: Try `frequency` method first
3. **For complex watermarks**: Use `adaptive` method (combines multiple techniques)
4. **Low contrast watermarks**: May require manual mask creation
5. **High quality images**: Generally produce better removal results

## Limitations

- Semi-transparent watermarks work better than solid ones
- Complex backgrounds may make detection more difficult
- Very large or very small watermarks may need parameter tuning
- Some artifacts may remain depending on watermark complexity

## Troubleshooting

If results are not satisfactory:

1. Try different methods (`-m inpaint`, `-m frequency`, `-m adaptive`)
2. Check the detected mask using `--show` option
3. For better detection, ensure good contrast between watermark and background
4. Consider preprocessing the image (contrast enhancement, noise reduction)

## Dependencies

- OpenCV (cv2) - Image processing
- NumPy - Numerical operations
- Matplotlib - Visualization
- SciPy - Scientific computing
- scikit-image - Additional image processing
- Pillow - Image I/O support

## License

This tool is provided for educational and research purposes. Please respect copyright laws and only remove watermarks from images you own or have permission to modify.