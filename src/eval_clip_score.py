import os
import sys

paths = ['../dalle-mini/src/']
for path in paths:
    if os.path.abspath(path) not in sys.path:
        sys.path.insert(0, os.path.abspath(path))
        # print(os.path.abspath(path))

import pandas as pd
import numpy as np
from PIL import Image
from IPython.display import display
import collections.abc as container_abcs
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from glob import glob
import argparse

import torch
from torchvision import transforms

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

from dalle_mini.text import TextNormalizer

from transformers import CLIPProcessor, CLIPModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--val_path', type=str, default="../data/data40k/validation.txt")
    parser.add_argument('--data_dir', type=str, default="../data/data40k/")
    parser.add_argument('--samples_dir', type=str, default="/home/andrewbai/glide_samples/")
    args = parser.parse_args()
    
    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    
    text_normalizer = TextNormalizer()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    df = pd.read_csv(args.val_path, sep=' ')
    
    all_logits = []
    
    sample_indices = [int(i) for i in os.listdir(os.path.join(args.samples_dir, args.ckpt))]
    
    for i in tqdm(sample_indices, leave=False):    
        prompt_path = os.path.join(args.data_dir, df["captions"].iloc[i])
        with open(prompt_path) as f:
            prompt = text_normalizer(f.readlines()[0])
        
        image_paths = glob(os.path.join(args.samples_dir, args.ckpt, str(i), '*.jpg'))
        images = [Image.open(image_path) for image_path in image_paths]
        
        with torch.no_grad():
            inputs = processor(text=[prompt], images=images, return_tensors="pt", 
                               padding='max_length', max_length=77, truncation=True)
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            logits_per_image = outputs.logits_per_image
            logits = logits_per_image.cpu().numpy().flatten()

        all_logits.append(logits)

    all_logits = np.stack(all_logits, axis=0)
    print(args.ckpt)
    print(f"the average max score is: {all_logits.max(1).mean()}")
    print(f"the average mean score is: {all_logits.mean()}")
    
if __name__ == '__main__':
    main()