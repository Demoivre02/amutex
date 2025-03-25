import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import create_model
import os


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    """
    model = create_model()
    checkpoints = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoints['params'])
    model.to(device)
    model.eval()
    return model


def load_img(image_path):
    """
    Load and preprocess the input image.
    """
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.0  
    img = img.astype(np.float32)
    return img


def process_img(image_path, output_dir):
    """
    Enhance the input image and save the output to the specified directory.
    Returns the path to the enhanced image.
    """
    
    img = load_img(image_path)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    
    enhanced_img = enhanced_tensor.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().numpy()
    enhanced_img = (enhanced_img * 255.0).round().astype(np.uint8)  
    enhanced_image = Image.fromarray(enhanced_img)

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    input_filename = os.path.basename(image_path)
    output_filename = f"enhanced_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)

    
    enhanced_image.save(output_path)
    print(f"Enhanced image saved to: {output_path}")

    return output_path


if __name__ == '__main__':
    
    input_image_path = 'images/low00772.png'  
    output_directory = 'FLOL/output'  
    model_path = 'FLOL/weights/flolv2_UHDLL.pt'  

    
    model = load_model(model_path)

    
    output_image_path = process_img(input_image_path, output_directory)

    
    enhanced_image = Image.open(output_image_path)
    enhanced_image.show()