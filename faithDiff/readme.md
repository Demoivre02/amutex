# FaithDiff Image Enhancement

This script enhances an image using dark channel prior-based dehazing techniques. It estimates atmospheric light and transmission to restore image clarity and contrast.

## Installation

### Prerequisites
Ensure you have Python installed (preferably Python 3.8 or later). You will also need `pip` to install dependencies.

### Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install numpy opencv-python
```

## Running the Script

### Steps
1. **Prepare Input Image:**
   - Place the image you want to enhance inside the `faithDiff/test/` directory.
   - Modify `input_image_path` in the script to point to your image file.

2. **Execute the Script:**
   Run the following command:
   
   ```bash
   python script.py 
   ```

   Ensure the script filename is correct when running the command.

3. **Output Image:**
   - The enhanced image will be saved inside the `faithDiff/output/` directory as `enhanced_image.jpg`.
   - The script will also display both the original and enhanced images.

## File Structure
```
.
├── script.py               # Main script
├── faithDiff/              # Main directory
│   ├── test/               # Directory for input images
│   │   ├── newyork.png     # Sample input image
│   ├── output/             # Directory for processed images
└── README.md               # This file
```

## Notes
- The script applies atmospheric light estimation and transmission refinement to improve image clarity.
- The processed image is automatically saved and displayed.

## License
This project is open-source and can be modified as needed.

