from imageai.Detection import ObjectDetection
import os
from PIL import Image, ImageDraw, ImageFont

def detect_objects(input_image_path, output_dir="imageAi/output", model_path="imageAi/retinanet_resnet50_fpn_coco-eeacb38b.pth"):
    """
    Perform object detection on an image using ImageAI RetinaNet and return the output file path and detected objects.

    Args:
        input_image_path (str): Path to the input image.
        output_dir (str): Directory to save output images.
        model_path (str): Path to the model file.

    Returns:
        tuple: A tuple containing:
            - output_image_path (str): Path to the output image with bounding boxes.
            - list_of_dicts (list): A list of dictionaries, where each dictionary represents a detected object.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()

    
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Image file not found: {input_image_path}")

    
    print("Performing object detection...")
    output_image_path = os.path.join(output_dir, "output_image.jpg")
    detections = detector.detectObjectsFromImage(
        input_image=input_image_path,
        output_image_path=output_image_path,
        minimum_percentage_probability=10
    )

    
    detected_image = Image.open(output_image_path)
    draw = ImageDraw.Draw(detected_image)

    
    box_thickness = 5  
    box_color = (255, 0, 0)  

    text_color = (0, 255, 0)  
    font_path = "arialbd.ttf"  
    font_size = 15  
    font = ImageFont.truetype(font_path, font_size)  

    
    list_of_dicts = []

    
    for obj in detections:
        box_points = obj["box_points"]  
        detected_object = {
            "category": obj['name'],  
            "coordinates": box_points,  
            "confidence": obj['percentage_probability']  
        }
        list_of_dicts.append(detected_object)

        
        draw.rectangle(
            [(box_points[0], box_points[1]), (box_points[2], box_points[3])],
            outline=box_color,
            width=box_thickness
        )

        
        label = f"{obj['name']}: {obj['percentage_probability']:.2f}%"
        draw.text(
            (box_points[0], box_points[1] - 20),  
            label,
            fill=text_color,  
            font=font  
        )

    
    customized_output_path = os.path.join(output_dir, "custom_output_image.jpg")
    detected_image.save(customized_output_path)

    
    if os.path.exists(output_image_path):
        os.remove(output_image_path)

    print("Detection complete!")
    return customized_output_path, list_of_dicts


if __name__ == "__main__":
    input_image_path = "images/hippo.jpg"  
    output_image_path, detected_objects = detect_objects(input_image_path)

    
    print(f"Output image saved to: {output_image_path}")

    
    print("Detected objects:")
    for obj in detected_objects:
        print(obj)  