Here’s a concise and clear `README.md` for your code:

---

# Object Detection with ImageAI and RetinaNet

This Python script performs object detection on an image using the **ImageAI** library and the **RetinaNet** model. It detects objects in the input image, draws bounding boxes around them, and saves the output image with annotations. It also returns a list of detected objects with their categories, coordinates, and confidence scores.

---

## **Requirements**

1. **Python 3.x**
2. **ImageAI** library:
   ```bash
   pip install imageai
   ```
3. **Pillow** library (for image processing):
   ```bash
   pip install pillow
   ```

---

## **Download the Pre-trained Model**

The RetinaNet model file (`retinanet_resnet50_fpn_coco-eeacb38b.pth`) is required for object detection. Download it from the following link:

📥 **Download Model**: [retinanet_resnet50_fpn_coco-eeacb38b.pth](https://drive.google.com/file/d/1MJHLtV6q-pVJjIjDwU0rNd5BX4aU4cC1/view?usp=drive_link)

After downloading, place the model file in the `imageAi/` directory.

---

## **How to Use**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Demoivre02/amutex
   cd amutex
   ```

2. **Run the Script**:
   Update the `input_image_path` variable in the script to point to your input image. Then run:
   ```bash
   python detect_objects.py
   ```

3. **Output**:
   - The script will save the output image with bounding boxes to `imageAi/output/custom_output_image.jpg`.
   - It will also print a list of detected objects with their categories, coordinates, and confidence scores.

---

## **Example**

```python
if __name__ == "__main__":
    input_image_path = "images/hippo.jpg"  # Path to the input image
    output_image_path, detected_objects = detect_objects(input_image_path)

    print(f"Output image saved to: {output_image_path}")
    print("Detected objects:")
    for obj in detected_objects:
        print(obj)
```

---

## **Output Format**

The script returns:
1. **Output Image Path**: Path to the annotated image.
2. **Detected Objects**: A list of dictionaries, where each dictionary contains:
   - `category`: The detected object's category (e.g., "person", "car").
   - `coordinates`: Bounding box coordinates as `[x1, y1, x2, y2]`.
   - `confidence`: Confidence score of the detection (percentage).

Example output:
```plaintext
Output image saved to: imageAi/output/custom_output_image.jpg
Detected objects:
{'category': 'person', 'coordinates': [100, 150, 200, 300], 'confidence': 95.0}
{'category': 'car', 'coordinates': [250, 100, 400, 200], 'confidence': 89.0}
```

---

## **Customization**

- **Input Image**: Update the `input_image_path` variable to point to your image.
- **Output Directory**: Change the `output_dir` parameter in the `detect_objects` function to save output images in a different directory.
- **Confidence Threshold**: Adjust the `minimum_percentage_probability` parameter in the `detectObjectsFromImage` function to filter detections by confidence.

---

## **Notes**

- Ensure the model file (`retinanet_resnet50_fpn_coco-eeacb38b.pth`) is placed in the `imageAi/` directory.
- The script uses the `arialbd.ttf` font for annotations. If the font is not available, replace it with a valid font path.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
