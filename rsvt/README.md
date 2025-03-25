# Image Enhancement Tool

This tool implements a dehazing algorithm to enhance images by reducing atmospheric haze and improving visibility. The algorithm is based on the dark channel prior method for single image dehazing.

## Features

- Dehazing algorithm to remove atmospheric haze from images
- Color enhancement in HSV space
- Combination of dehazing and color enhancement for optimal results
- Automatic output directory creation
- Visual comparison of original and enhanced images

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install the required packages with:
```
pip install opencv-python numpy
```

## Usage

### Basic Usage
```python
from image_enhancer import enhance_image

output_path = enhance_image(input_image_path, output_directory)
```

### Command Line Execution
Run the script directly to process a sample image:
```bash
python script.py
```

### Parameters
- `image_path`: Path to the input image file
- `output_dir`: Directory where the enhanced image will be saved

The enhanced image will be saved with the prefix "enhanced_" followed by the original filename.



## Example

```python
enhance_image('input.jpg', 'output/')
```

This will create an enhanced version of `input.jpg` in the `output/` directory.

## Output

The function returns the path to the enhanced image and displays both original and enhanced images for comparison when run directly.

