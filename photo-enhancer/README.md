# Image Enhancement Tool

## Overview
This tool enhances images using various enhancement methods, including GFPGAN, RestoreFormer, and CodeFormer. It provides options for background enhancement and upscaling.

## Important Setup Note
Before using this tool, you must download the required model weights:

1. **Download these model files**:
   - [detection_Resnet50_Final.pth](https://drive.google.com/file/d/13Eu2HfjbLDXbIbCQObiBoSgZ5Du9Femn/view?usp=drive_link)
   - [parsing_parsenet.pth](https://drive.google.com/file/d/1tpQxvfCyhNpcUxFDyrU7aPQXZpsFn-Fp/view?usp=drive_link)

2. **Place the downloaded files** in:
   ```
   gfpgan/weights/
   ```
   (Create the folder if it doesn't exist)

These files are not included in the repository due to their large size.

## Requirements
- Python 3.7+
- Required libraries:
  - `Pillow`
  - `numpy`
  - `torchvision`
  - `argparse`
  - `enhancer` (custom module, ensure it is available)

## Installation
1. First download the model weights as described above
2. Then install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the script with the required arguments:
```sh
python script.py --method <method> --image_path <input_image> --output_path <output_image> [--background_enhance] [--upscale <factor>]
```

### Arguments:
- `--method`: Specify the enhancement method (`gfpgan`, `RestoreFormer`, `codeformer`)
- `--image_path`: Path to the input image
- `--output_path`: Path to save the enhanced image
- `--background_enhance`: Enable background enhancement (default: enabled)
- `--upscale`: Specify the upscaling factor (2 or 4)

### Example:
```sh
python script.py --method gfpgan --image_path samples/input.jpg --output_path output/enhanced.jpg --upscale 2
```

## Folder Structure
```
photo-enhancer/
├── gfpgan/
│   └── weights/            # Place downloaded .pth files here
├── output/                 # Enhanced images will be saved here
├── samples/                # Sample input images
├── enhancer.py             # Core enhancement module
├── README.md
├── requirements.txt
└── script.py               # Main execution script
```
