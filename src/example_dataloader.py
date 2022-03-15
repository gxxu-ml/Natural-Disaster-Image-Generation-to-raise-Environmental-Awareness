import argparse
from tqdm import tqdm
from functools import partial

import torch
import torchvision.transforms as transforms
from vqgan_jax.modeling_flax_vqgan import VQModel

from data import get_data

parser = argparse.ArgumentParser()

# dataloader args
parser.add_argument('--data-dir', type=str, default="../data")
parser.add_argument('--train-filename', type=str, default="../data/train.txt")
parser.add_argument('--valid-filename', type=str, default="../data/validation.txt")
parser.add_argument('--img-key', type=str, default="images")
parser.add_argument('--caption-key', type=str, default="captions")
parser.add_argument("--csv-separator", type=str, default=" ")
parser.add_argument('--train-batch_size', type=int, default=4)
parser.add_argument('--valid-batch-size', type=int, default=4)
# transforms.ToTensor() changes input format from H x W x C to C x H x W
# don't use below argument if input format required is C x H x W
# use it to convert back from C x H x W
# vqgan_jax requires H x W x C format, so set it to run this script
parser.add_argument('--permute', action='store_true')

args = parser.parse_args()
assert args.permute, 'args.permute should be set in this script, see comments'


VQGAN_REPO, VQGAN_COMMIT_ID = (
    "dalle-mini/vqgan_imagenet_f16_16384",
    "85eb5d3b51a1c62a0cc8f4ccdee9882c0d0bd384",
)

vqgan = VQModel.from_pretrained(VQGAN_REPO)
vqgan_params = vqgan.params

def p_encode(batch, params):
    _, indices = vqgan.encode(batch, params=params)
    return indices

def encode_dataset(dataloader):
    for idx, (images, captions) in enumerate(tqdm(dataloader)):
        images = images.numpy()
        encoded = p_encode(images, vqgan_params)
        print(encoded.shape, captions.shape)
        print('breaking out of loop')
        break

# dummy transformations
image_resolution = 256
train_transform = transforms.Compose(
    [transforms.Resize(image_resolution),
     transforms.RandomCrop(image_resolution),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)
valid_transform = transforms.Compose(
    [transforms.Resize(image_resolution),
     transforms.CenterCrop(image_resolution),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

# tokenize=None will use default tokenizer from min-dalle repo
# pass tokenizer if a different one is required
data = get_data(args, (train_transform, valid_transform), 'dalle', tokenize=None)
data.setup()
train_dataloader = data.train_dataloader()
encode_dataset(train_dataloader)
