from PIL import Image
import numpy as np
from enhancer import Enhancer
import argparse
import os
from torchvision.transforms.functional import rgb_to_grayscale


def main(method, image_path, output_path, background_enhancement, upscale):
    """
    Enhance the input image using the specified method and save the output.
    Returns the path to the enhanced image.
    """
    
    enhancer = Enhancer(method=method, background_enhancement=background_enhancement, upscale=upscale)

    
    try:
        image = np.array(Image.open(image_path))
    except Exception as e:
        raise ValueError(f"Failed to load the image from {image_path}. Error: {e}")

    
    restored_image = enhancer.enhance(image)

    
    final_image = Image.fromarray(restored_image)
    
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    final_image.save(output_path)
    print(f"Enhanced image saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Enhance an image using a specified method.")
    
    parser.add_argument("--method", type=str, required=True, 
                        help="Specify the enhance method (gfpgan, RestoreFormer, codeformer).")
    
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Specify the path to the input image.")
    
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Specify the path to save the enhanced image.")
    
    parser.add_argument("--background_enhancement", action="store_true", 
                        help="Enable background enhancement.", default=True)
    
    parser.add_argument("--upscale", type=int, 
                        help="Specify the enhancement scale (2, 4).")
    
    args = parser.parse_args()
    
    try:
        
        output_image_path = main(args.method, args.image_path, args.output_path, args.background_enhancement, args.upscale)
        print(f"Enhanced image saved successfully: {output_image_path}")
    except Exception as e:
        print(f"Error: {e}")