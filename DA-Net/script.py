import os
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from PIL import Image
import numpy as np
from DA_Net import DA_Net_t  

def load_model(model_path, device):
    """
    Load the pre-trained DA-Net model.
    """
    net = DA_Net_t().to(device)
    net = nn.DataParallel(net)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    return net

def preprocess_image(image_path):
    """
    Preprocess the input image for the model.
    """
    image = Image.open(image_path).convert("RGB")
    transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  
    return image

def enhance_image(image_path, model, device):
    """
    Enhance the input image using the DA-Net model.
    """
    
    input_image = preprocess_image(image_path).to(device)

    
    with torch.no_grad():
        output = model(input_image)

    
    output = torch.squeeze(output.clamp(0, 1).cpu()).permute(1, 2, 0).numpy()
    output = (output * 255).astype(np.uint8)
    return output

def save_image(image, output_path):
    """
    Save the enhanced image to the specified path.
    """
    Image.fromarray(image).save(output_path)

def main():
    
    input_image_path = "DA-Net/img/haze.png"  
    output_dir = "DA-Net/output"  
    model_path = "DA-Net/trained_models/DA-Net_RSID_146.pk"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    model = load_model(model_path, device)

    
    enhanced_image = enhance_image(input_image_path, model, device)

    
    output_image_path = os.path.join(output_dir, "enhanced_image.jpg")
    save_image(enhanced_image, output_image_path)

    print(f"Enhanced image saved to: {output_image_path}")

if __name__ == "__main__":
    main()