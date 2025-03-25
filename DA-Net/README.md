# DA-Net Image Enhancement

This script utilizes the DA-Net model to enhance image quality by removing haze and improving clarity. The model is loaded from a pre-trained checkpoint, processes an input image, and saves the enhanced version.

## Installation

### Prerequisites
Ensure you have Python installed (preferably Python 3.8 or later). You will also need `pip` for installing dependencies.

### Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install torch torchvision pillow numpy
```

## Running the Script

### Steps
1. **Prepare Input Image:**
   - Place the image you want to enhance inside the `DA-Net/img/` directory.
   - Modify `input_image_path` in the script to point to your image file.

2. **Execute the Script:**
   Run the following command:
   
   ```bash
   python script.py
   ```

   Ensure the script filename is correct when running the command.

3. **Output Image:**
   - The enhanced image will be saved inside the `DA-Net/output/` directory as `enhanced_image.jpg`.
   - The script will print the saved image path upon completion.

## File Structure
```
.
├── script.py              # Main script
├── DA-Net/                # Main directory
│   ├── img/               # Directory for input images
│   │   ├── haze.png       # Sample input image
│   ├── output/            # Directory for processed images
│   ├── trained_models/    # Directory for storing trained model
│   │   ├── DA-Net_RSID_146.pk  # Pre-trained model file
└── README.md              # This file
```

## Notes
- The script automatically loads the DA-Net model from the specified checkpoint.
- It applies haze removal and image enhancement before saving the output.
- Ensure you have a GPU available for optimal performance, though the script will work on a CPU as well.

## License
This project is open-source and can be modified as needed.

