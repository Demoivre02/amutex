# Object Detection with SAHI and Hugging Face Models

This project utilizes the **SAHI (Slicing Aided Hyper Inference)** library for object detection using YOLOv8 and Hugging Face models. The implementation includes downloading a pre-trained model, performing inference on images, and visualizing the predictions.

## 🚀 Features
- Uses **SAHI** for sliced inference on large images.
- Supports **YOLOv8** and **Hugging Face RT-DETR** models.
- Exports visualized detection results.
- Runs on **CPU** or **GPU**.

---

## 📌 Installation
Before running the project, install the required dependencies:

```bash
pip install -U sahi transformers timm ultralytics
```

For Hugging Face model access, **create an account** on [Hugging Face](https://huggingface.co/) and generate a **read access token**.

To authenticate, set up the token:
```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token"
```

---

## 📂 Project Structure
```
sahi_yolo/
│── images/                    # Folder for input images
│   ├── graffiti-vandals.jpg    # Example image
│── output/                     # Folder for storing predictions
│── yolov8n.pt                  # YOLOv8 model file (if using YOLO)
│── script.py                   # Main script
│── README.md                   # This documentation
```

---

## 🔧 Usage

### **1️⃣ Run Inference with YOLOv8**
Run the detection script using YOLO:
```bash
python script.py
```

Ensure `yolov8n.pt` exists in the project directory. If missing, download it manually:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8/yolov8n.pt
```

### **2️⃣ Run Inference with Hugging Face Model**
Modify the script to use **PekingU RT-DETR** model:
```python
model_path = "PekingU/rtdetr_v2_r18vd"  # Change to a larger model if needed
```
Run the script:
```bash
python script.py
```

---

## 🖼️ Visualizing Results
The processed images are saved in the `output/` folder. You can view them using:
```python
from IPython.display import display, Image

output_image_path = "sahi_yolo/output/prediction_visual.png"
display(Image(output_image_path))
```

---

## ⚙️ Troubleshooting
### **Model Not Found?**
- Ensure the model exists locally or is correctly specified.
- If using a Hugging Face model, verify your token.

### **Image Not Found?**
- Place the input image in `sahi_yolo/images/`.

### **CUDA Error?**
- If using a GPU, ensure PyTorch detects it:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
- If `False`, install the correct **PyTorch + CUDA** version from: [PyTorch](https://pytorch.org/).

---

## 📜 License
This project is open-source and free to use.

---

## 🤝 Contributing
Feel free to submit issues or pull requests!

