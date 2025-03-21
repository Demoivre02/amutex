import cv2
import numpy as np
import os

def detect_objects(image_path, output_dir="output", model_weights="openCV/mobilenet_iter_73000.caffemodel", model_config="openCV/deploy.prototxt"):
    """
    Perform object detection on an image using OpenCV's DNN module and return the output file path and detected objects.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output image.
        model_weights (str): Path to the model weights file.
        model_config (str): Path to the model configuration file.

    Returns:
        tuple: A tuple containing:
            - output_image_path (str): Path to the output image with bounding boxes.
            - list_of_dicts (list): A list of dictionaries, where each dictionary represents a detected object.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

    
    class_names = [
        "background", "aeroplane", "bicycle", "animal", "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    
    (h, w) = image.shape[:2]

    
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)

    
    detections = net.forward()

    
    list_of_dicts = []

    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            class_name = class_names[class_id]

            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            
            detected_object = {
                "category": class_name,  
                "coordinates": [startX, startY, endX, endY],  
                "confidence": float(confidence)  
            }
            list_of_dicts.append(detected_object)

            
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    display_width = 800
    aspect_ratio = display_width / w
    display_height = int(h * aspect_ratio)
    resized_image = cv2.resize(image, (display_width, display_height))

    
    cv2.imshow("Output", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    output_image_path = os.path.join(output_dir, "output.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Output saved to {output_image_path}")

    return output_image_path, list_of_dicts


if __name__ == "__main__":
    image_path = "images/dump.jpg"  
    output_image_path, detected_objects = detect_objects(image_path)

    
    print(f"Output image saved to: {output_image_path}")

    
    print("Detected objects:")
    for obj in detected_objects:
        print(obj)  