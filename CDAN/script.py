import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image
import os

from cdan import CDAN  


def load_model():
    """
    Load the pre-trained CDAN model from Hugging Face Hub.
    """
    model_repo = "hossshakiba/CDAN"
    model_path = hf_hub_download(repo_id=model_repo, filename="CDAN.pt")
    
    model = CDAN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


preprocess = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Resize((400, 600)),  
])


def enhance_contrast(images, contrast_factor=1.5):
    """
    Enhance the contrast of the input images.
    """
    if images.max() > 1.0:
        images = images / 255.0
    
    mean_intensity = images.mean(dim=(2, 3), keepdim=True)
    enhanced_images = (images - mean_intensity) * contrast_factor + mean_intensity
    enhanced_images = torch.clamp(enhanced_images, 0.0, 1.0)
    
    return enhanced_images


def enhance_color(images, saturation_factor=1.5):
    """
    Enhance the color saturation of the input images.
    """
    if images.max() > 1.0:
        images = images / 255.0
    
    grayscale = 0.2989 * images[:, 0, :, :] + 0.5870 * images[:, 1, :, :] + 0.1140 * images[:, 2, :, :]
    grayscale = grayscale.unsqueeze(1)  
    
    enhanced_images = grayscale + saturation_factor * (images - grayscale)
    enhanced_images = torch.clamp(enhanced_images, 0.0, 1.0)
    
    return enhanced_images


def process_image(input_image_path, output_dir):
    """
    Enhance the input image and save the output to the specified directory.
    Returns the path to the enhanced image.
    """
    
    input_image = Image.open(input_image_path).convert('RGB')
    
    
    input_tensor = preprocess(input_image).unsqueeze(0)  
    
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    
    output_tensor = enhance_contrast(output_tensor, contrast_factor=1.12)
    output_tensor = enhance_color(output_tensor, saturation_factor=1.35)
    
    
    output_tensor = output_tensor.squeeze(0).clamp(0, 1)  
    output_image = transforms.ToPILImage()(output_tensor)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    input_filename = os.path.basename(input_image_path)
    output_filename = f"enhanced_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    
    output_image.save(output_path)
    print(f"Enhanced image saved to: {output_path}")
    
    return output_path


if __name__ == '__main__':
    
    input_image_path = 'images/example3.png'  
    output_directory = 'CDAN/output'  

    
    model = load_model()

    
    output_image_path = process_image(input_image_path, output_directory)

    
    output_image = Image.open(output_image_path)
    output_image.show()