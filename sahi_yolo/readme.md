# SAHI YOLO Object Detection

## Overview
This project leverages the **SAHI (Slicing Aided Hyper Inference) framework** and **YOLOv8** for object detection. It processes images using slicing-based inference to improve detection accuracy.

## Installation
To set up the environment and install the necessary dependencies, run the following commands:

```sh

# Install Python dependencies
pip install -U sahi ultralytics
pip install transformers timm
```

## Model Setup
You need to download the YOLOv8 model before running the script. You can do this using:

```sh
wget -P sahi_yolo/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Alternatively, you can download the model from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and place it in the `sahi_yolo/` directory.

## Usage
Run the detection script using:

```sh
python script.py
```

The script:
1. Loads the YOLOv8 model.
2. Reads an input image from the `images/` directory.
3. Performs sliced object detection for improved accuracy.
4. Saves the results in the `sahi_yolo/output/` directory.
5. Displays the output image.

## File Structure
```
project_root/
│── sahi_yolo/
│   ├── yolov8n.pt
│   ├── output/
│   ├── images/
│       ├── zoo.jpg
│── script.py
```

## Troubleshooting
- If the model file is missing, ensure `yolov8n.pt` is in the `sahi_yolo/` directory.
- If an image is missing, confirm the file path in the `images/` folder.
- If the output image does not appear, check `sahi_yolo/output/` for errors.

## Acknowledgments
- [SAHI Documentation](https://github.com/obss/sahi)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---
Happy detecting! 🚀

