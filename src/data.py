import os
from PIL import Image
from functools import partial
import numpy as np

import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return T.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

class RandomCropLongEdge(object):
    """Crops the given PIL Image on the long edge with a random start point.
    Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] 
              else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
              else np.random.randint(low=0,high=img.size[1] - size[1]))
        return T.functional.crop(img, j, i, size[1], size[0])

    def __repr__(self):
        return self.__class__.__name__


# TODO make data_dir configurable
# TODO make sep configurable for dalle, glide
class CustomDataset(Dataset):
    """
    Dataset to load images,captions from image-caption mapping file
    """
    def __init__(self, input_filename, transforms, img_key, caption_key,
                 tokenize=None, data_dir="../data", sep=" ", permute=False):
        super().__init__()
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = None
        if caption_key in df.columns:
            self.captions = df[caption_key].tolist()
        self.data_dir = data_dir
        self.tokenize = tokenize

        self.transforms = transforms
        assert self.transforms is not None
        # T.ToTensor() -> C x H x W format
        # in case need H x W x C format back
        self.permute = permute

        self.post_init()

    def post_init(self):
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(
            os.path.join(self.data_dir, str(self.images[idx]))).convert("RGB"))
        if self.permute:
            image = image.permute(1, 2, 0)

        if self.captions:
            text = open(os.path.join(self.data_dir, str(self.captions[idx])), 'r').read()
            text = self.tokenize(text)
            if not torch.is_tensor(text):
                text = torch.tensor(text)
            return image, text.squeeze()
        return image

class GeneralCustomDataset(CustomDataset):
    def __init__(self, input_filename, transforms, img_key, caption_key,
                 tokenize=None, data_dir="../data", sep=" ", permute=False,
                 shard=0, num_shards=1):
        super().__init__(input_filename, transforms, img_key, caption_key,
                         tokenize, data_dir, sep, permute)
        
        # sharding data for DDP
        l = len(self.images)
        start_index = int(shard / num_shards * l)
        end_index = int((shard + 1) / num_shards * l)
        self.images = self.images[start_index:end_index]
        self.captions = self.captions[start_index:end_index]
        # print(start_index, end_index)
        
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_dir, str(self.images[idx]))).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        if self.permute:
            image = image.permute(1, 2, 0)

        if self.captions:
            text = open(os.path.join(self.data_dir, str(self.captions[idx])), 'r').read()
            cond = self.tokenize(text) # returns a dict
            cond = {k: v if torch.is_tensor(v) else torch.tensor(v)
                    for k, v in cond.items()}
            return image, cond
        else:
            return image

# TODO hardcoded image key
class ImageDataset(CustomDataset):
    """
    Dataset to load images from image-caption mapping file
    """
    def post_init(self):
        self.transforms = eval(self.transforms)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.data_dir, str(self.images[idx])))
        return {"image": self.transforms(image).permute(1, 2, 0)}


class CustomDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_batch_size: int = 2,
                 valid_batch_size: int = 32,
                 train_filename = None,
                 valid_filename = None,
                 num_workers: int = 8,
                 test_filename = None,
                 train_setup_fn = None,
                 valid_setup_fn = None,
                 collate_fn = None,
                 pin_memory = False):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_filename = train_filename
        self.valid_filename = valid_filename

        self.dataset_configs = dict()
        self.setup_fn = dict()
        if train_filename is not None:
            self.dataset_configs["train"] = train_filename
            self.train_dataloader = self._train_dataloader
            self.setup_fn["train"] = train_setup_fn
        if valid_filename is not None:
            self.dataset_configs["validation"] = valid_filename
            self.valid_dataloader = self._valid_dataloader
            self.setup_fn["validation"] = valid_setup_fn
        if test_filename is not None:
            self.dataset_configs["test"] = test_filename
            self.test_dataloader = self._test_dataloader
            self.setup_fn["test"] = valid_setup_fn

        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.datasets = dict(
            (k, self.setup_fn[k](self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"],
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)

    def _valid_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"],
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)


# TODO train and validation transforms
class ImageDataModule(CustomDataModule):
    def __init__(self, batch_size, train=None, validation=None,
                 test=None, wrap=False, num_workers=None):
        from vqgan.main import instantiate_from_config
        from vqgan.taming.data.utils import custom_collate
        super().__init__(
                 train_batch_size=batch_size,
                 valid_batch_size=batch_size,
                 train_filename=train,
                 valid_filename=validation,
                 num_workers=num_workers,
                 test_filename=test,
                 train_setup_fn=instantiate_from_config,
                 valid_setup_fn=instantiate_from_config,
                 collate_fn=custom_collate,
                 pin_memory = False)

        self.batch_size = batch_size
        self.val_dataloader = self._valid_dataloader
        self.instantiate_from_config_ = instantiate_from_config

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            self.instantiate_from_config_(data_cfg)


def get_clip_data(args, preprocess_fns, **kwargs):
    from open_clip.clip.clip import tokenize
    from open_clip.training.data import DataInfo

    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    def get_custom_dataset(args, preprocess_fn, is_train):
        input_filename = args.train_data if is_train else args.val_data
        assert input_filename

        dataset = CustomDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            tokenize=tokenize,
            sep=args.csv_separator)
        num_samples = len(dataset)
        sampler = DistributedSampler(dataset) if args.distributed and is_train else None
        shuffle = is_train and sampler is None
   
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
        dataloader.num_samples = num_samples
        dataloader.num_batches = len(dataloader)
   
        return DataInfo(dataloader, sampler)

    if args.train_data:
        data["train"] = get_custom_dataset(args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_custom_dataset(args, preprocess_val, is_train=False)
    return data

# TODO shuffle arg for dalle dataloader
def get_dalle_data(args, preprocess_fns, tokenize, **kwargs):
    preprocess_train, preprocess_val = preprocess_fns
    if tokenize is None:
        from dalle.dalle.models.tokenizer import build_tokenizer
        # TODO configurable tokenizer path
        _tokenizer = build_tokenizer('dalle/tokenizer',
                                     lowercase=True,
                                     dropout=None)
        def _tokenize(x):
            return _tokenizer.encode(x).ids
        tokenize = _tokenize

    def setup_fn(transforms, input_filename):
        return CustomDataset(input_filename, transforms, args.img_key, args.caption_key,
                             tokenize, args.data_dir, args.csv_separator, args.permute)

    return CustomDataModule(args.train_batch_size, args.valid_batch_size,
                            args.train_filename, args.valid_filename,
                            train_setup_fn=partial(setup_fn, preprocess_train),
                            valid_setup_fn=partial(setup_fn, preprocess_val), pin_memory=True)

def get_glide_data(args, preprocess_fns, tokenize, shard=0, num_shards=1, **kwargs):
    preprocess_train, preprocess_val = preprocess_fns
    
    def setup_fn(transforms, input_filename):
        return GeneralCustomDataset(input_filename, transforms, args.img_key, args.caption_key,
                                    tokenize, args.data_dir, args.csv_separator, args.permute,
                                    shard, num_shards)

    def collate_fn(data):
        batch, conds = zip(*data)
        batch = torch.stack(batch, dim=0)
        ks = conds[0].keys()
        conds = {k: torch.stack([cond[k] for cond in conds], dim=0) 
                 for k in ks}
        return batch, conds
    
    return CustomDataModule(args.batch_size, args.batch_size,
                            args.train_filename, args.valid_filename,
                            train_setup_fn=partial(setup_fn, preprocess_train),
                            valid_setup_fn=partial(setup_fn, preprocess_val), 
                            pin_memory=True, collate_fn=collate_fn)

def get_data(args, preprocess_fns, model_type, tokenize=None, **kwargs):
    if model_type == 'clip':
        return get_clip_data(args, preprocess_fns, **kwargs)
    elif model_type == 'dalle':
        # assert tokenize is not None, "dalle tokenizer is part of model config"
        return get_dalle_data(args, preprocess_fns, tokenize, **kwargs)
    elif model_type == 'glide':
        return get_glide_data(args, preprocess_fns, tokenize, **kwargs)
    else:
        print("either get_data method is not implemented or you shouldn't be using it")
        raise NotImplementedError
