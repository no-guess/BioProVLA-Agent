#!/usr/bin/env python3
"""
Offline image enhancer for LeRobot datasets
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
from lerobot.policies.smolvla.modeling_smolvla import VisualEnhancer

# Initialize enhancer
enhancer = VisualEnhancer(strength=0.6, gamma=0.9, learnable=False)
enhancer.eval()

def enhance_image(image_path, output_path):
    """Enhance a single image"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        enhanced = enhancer(tensor)
    
    enhanced_img = transforms.ToPILImage()(enhanced.squeeze(0))
    enhanced_img.save(output_path)

def enhance_directory(input_dir, output_dir):
    """Enhance all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            enhance_image(input_path, output_path)

if __name__ == "__main__":
    # Example usage
    input_dir = "path/to/your/dataset/images"
    output_dir = "path/to/enhanced/images"
    enhance_directory(input_dir, output_dir)