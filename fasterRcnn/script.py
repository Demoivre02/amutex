from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from IPython.display import display, Image
import os
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def detect_objects(image_path):
    """
    Perform object detection on an image using Faster R-CNN and return the output file path and detected objects.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: A tuple containing:
            - output_image_path (str): Path to the output image with bounding boxes.
            - list_of_dicts (list): A list of dictionaries, where each dictionary represents a detected object.
    """
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="torchvision",
        model=model,
        confidence_threshold=0.5,
        device="cpu",  
        load_at_init=True,
    )

    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=612,  
        slice_width=612,  
        overlap_height_ratio=0.2,  
        overlap_width_ratio=0.2,  
    )

    
    output_dir = "fasterRcnn/output"
    os.makedirs(output_dir, exist_ok=True)

    
    result.export_visuals(export_dir=output_dir)

    
    output_image_path = os.path.join(output_dir, "prediction_visual.png")

    
    if os.path.exists(output_image_path):
        display(Image(output_image_path))
    else:
        print("Output image not found. Check SAHI's export settings.")

    
    list_of_dicts = []
    for prediction in result.object_prediction_list:
        bbox = prediction.bbox
        detected_object = {
            "category": prediction.category.name,  
            "coordinates": [int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)],  
            "confidence": float(prediction.score.value)  
        }
        list_of_dicts.append(detected_object)

    
    return output_image_path, list_of_dicts


if __name__ == "__main__":
    image_path = "images/hippo.jpg"  
    output_image_path, detected_objects = detect_objects(image_path)

    
    print(f"Output image saved to: {output_image_path}")

    
    print("Detected objects:")
    for obj in detected_objects:
        print(obj)  