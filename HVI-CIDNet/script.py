import os
import numpy as np
import torch
from PIL import Image
from net.CIDNet import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
import json


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def from_pretrained(cls, pretrained_model_name_or_path: str):
    model_id = str(pretrained_model_name_or_path)

    
    config_file = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    config = None
    if config_file is not None:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

    
    model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors", repo_type="model")
    state_dict = sf.load_file(model_file)
    cls.load_state_dict(state_dict, strict=False)


eval_net = CIDNet().to(device)
eval_net.trans.gated = True
eval_net.trans.gated2 = True


def process_image(input_img_path, output_dir="output", model_path="Generalization", gamma=1.0, alpha_s=1.0, alpha_i=1.0):
    """
    Enhances the input image and saves the output to the specified directory.
    Returns the path to the enhanced image.
    """
    
    from_pretrained(eval_net, "Fediory/HVI-CIDNet-" + model_path)
    eval_net.eval()

    
    input_img = Image.open(input_img_path).convert('RGB')
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input_tensor = pil2tensor(input_img)

    
    factor = 8
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = F.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')

    
    with torch.no_grad():
        eval_net.trans.alpha_s = alpha_s
        eval_net.trans.alpha = alpha_i
        output_tensor = eval_net(input_tensor.to(device) ** gamma)

    
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_tensor = output_tensor[:, :, :h, :w]
    enhanced_img = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    output_filename = os.path.basename(input_img_path).split('.')[0] + "_enhanced.jpg"
    output_image_path = os.path.join(output_dir, output_filename)
    enhanced_img.save(output_image_path)

    return output_image_path

if __name__ == '__main__':
    
    input_image_path = 'images/example2.png'  
    output_dir = "HVI-CIDNet_Low-light-Image-Enhancement_/output"  

    
    output_image_path = process_image(input_image_path, output_dir)

    
    print(f"Enhanced image saved to {output_image_path}")