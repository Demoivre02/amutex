# CDAN Image Enhancement

This script uses the CDAN (Contrastive Domain Adaptation Network) model to enhance image contrast and color saturation. It downloads a pre-trained CDAN model from the Hugging Face Hub, processes an input image, and saves the enhanced version.

## Installation

### Prerequisites
Ensure you have Python installed (preferably Python 3.8 or later). You will also need `pip` for installing dependencies.

### Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install torch torchvision huggingface_hub pillow
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

   Ensure the script filename is correct when running the command.

3. **Output Image:**
   - The enhanced image will be saved inside the `CDAN/output/` directory with a prefix `enhanced_`.
   - The image will also be displayed automatically upon processing.

## File Structure
```
.
├── script.py              # Main script
├── images/                # Directory for input images
│   ├── example3.png       # Sample input image
├── CDAN/                  # Output directory
│   ├── output/            # Directory for processed images
└── README.md              # This file
```

## Notes
- The script automatically downloads the CDAN model from Hugging Face on first run.
- It applies both contrast and color enhancement before saving the output.

## License
This project is open-source and can be modified as needed.

