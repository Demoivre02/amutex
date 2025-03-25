# HVI-CIDNet Image Enhancement

This script utilizes the HVI-CIDNet model for enhancing images, specifically for low-light image enhancement. It downloads a pre-trained HVI-CIDNet model from the Hugging Face Hub, processes an input image, and saves the enhanced version.

## Installation

### Prerequisites
Ensure you have Python installed (preferably Python 3.8 or later). You will also need `pip` for installing dependencies.

### Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install torch torchvision pillow safetensors huggingface_hub numpy
```

## Running the Script

### Steps
1. **Prepare Input Image:**
   - Place the image you want to enhance inside the `images/` directory.
   - Modify `input_image_path` in the script to point to your image file.

2. **Execute the Script:**
   Run the following command:
   
   ```bash
   python script.py
   ```

3. **Output Image:**
   - The enhanced image will be saved inside the `output/` directory with a suffix `_enhanced.jpg`.
   - The script will print the path of the saved image upon successful execution.

## File Structure
```
.
├── script.py                     # Main script
├── images/                        # Directory for input images
│   ├── example2.png               # Sample input image
├── output/                        # Directory for processed images
└── README.md                      # This file
```

## Notes
- The script automatically downloads the HVI-CIDNet model from Hugging Face on first run.
- It applies image enhancement using configurable parameters such as `gamma`, `alpha_s`, and `alpha_i`.
- The model uses reflect padding to handle images of different sizes efficiently.

## License
This project is open-source and can be modified as needed.

