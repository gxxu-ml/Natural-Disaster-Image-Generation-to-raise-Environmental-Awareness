import sys
import os

paths = ['../guided-diffusion/']
for path in paths:
    if os.path.abspath(path) not in sys.path:
        sys.path.insert(0, os.path.abspath(path))
        # print(os.path.abspath(path))
        
import pandas as pd
import numpy as np
from PIL import Image
from IPython.display import display
import collections.abc as container_abcs
import argparse
from tqdm import trange
from mpi4py import MPI

import torch
from torchvision import transforms

from guided_diffusion import dist_util

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_step', type=int, default=11000)
    parser.add_argument('--ckpt_dir', type=str, default="/home/andrewbai/glide_logs/")
    parser.add_argument('--val_path', type=str, default="../data/data40k/validation.txt")
    parser.add_argument('--data_dir', type=str, default="../data/data40k/")
    parser.add_argument('--save_dir', type=str, default="/home/andrewbai/glide_samples/")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_test_samples', type=int, default=200)
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % dist_util.GPUS_PER_NODE}"
    else:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        device_index = MPI.COMM_WORLD.Get_rank() % dist_util.GPUS_PER_NODE
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_visible_devices[device_index]}"

    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    # print(device)

    ### Loading GLIDE

    # Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)

    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    if args.ckpt is None:
        print(f"Load pretrained model.")
        model.load_state_dict(load_checkpoint('base', device))
        args.ckpt = 'base'
    else:
        print(f"Load finetuned model from {args.ckpt}.")
        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt, f'model{args.ckpt_step:06d}.pt')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # print('total base parameters', sum(x.numel() for x in model.parameters()))

    guidance_scale = 3.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def generate(prompt: str, rank: int):
        with torch.no_grad():
            # Create the text tokens to feed to the model.
            tokens = model.tokenizer.encode(prompt)
            tokens, mask = model.tokenizer.padded_tokens_and_mask(
                tokens, options['text_ctx']
            )
            # Create the classifier-free guidance tokens (empty)
            full_batch_size = args.batch_size * 2
            uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
                [], options['text_ctx']
            )
            # Pack the tokens together into model kwargs.
            model_kwargs = dict(
                tokens=torch.tensor(
                    [tokens] * args.batch_size + [uncond_tokens] * args.batch_size, device=device
                ),
                mask=torch.tensor(
                    [mask] * args.batch_size + [uncond_mask] * args.batch_size,
                    dtype=torch.bool,
                    device=device,
                ),
            )
            # Sample from the base model.
            model.del_cache()
            samples = diffusion.p_sample_loop(
                model_fn,
                (full_batch_size, 3, options["image_size"], options["image_size"]),
                device=device,
                clip_denoised=True,
                progress=(rank == 0),
                model_kwargs=model_kwargs,
                cond_fn=None,
            )[:args.batch_size]
            model.del_cache()
            samples = samples.cpu()

        images = [transforms.ToPILImage()(sample) for sample in samples]

        return images

    shard = MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()
    start_index = int(shard / num_shards * args.num_test_samples)
    end_index = int((shard + 1) / num_shards * args.num_test_samples)

    df = pd.read_csv(args.val_path, sep=' ')
    for i in trange(start_index, end_index, leave=False) if shard == 0 else range(start_index, end_index):
        os.makedirs(os.path.join(args.save_dir, args.ckpt, str(i)), exist_ok=True)
        prompt_path = os.path.join(args.data_dir, df["captions"].iloc[i]) 
        with open(prompt_path) as f:
            prompt = f.readlines()[0]

        #generate images and logits
        images = generate(prompt, shard)
        for j, image in enumerate(images):
            image.save(os.path.join(args.save_dir, args.ckpt, str(i), f"{j}.jpg"))

if __name__ == '__main__':
    main()
