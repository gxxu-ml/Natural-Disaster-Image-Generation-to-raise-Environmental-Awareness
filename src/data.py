import os
from PIL import Image

import torchvision.transforms as transforms
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from open_clip.training.data import CsvDataset
from vqgan.taming.data.utils import custom_collate
from vqgan.main import instantiate_from_config


# TODO make data_dir configurable
# TODO make sep configurable for dalle, glide
class CustomDataset(CsvDataset):
    """
    Dataset to load images,captions from image-caption mapping file
    """
    def __init__(self, input_filename, transform, img_key,
                    caption_key, tokenize=None, data_dir="../data", sep=" "):
        # super().__init__(os.path.abspath(input_filename), transform, img_key, caption_key, sep)
        super().__init__(input_filename, transform, img_key, caption_key, sep)
        self.data_dir = data_dir
        self.tokenize = tokenize
        # TODO remove this block
        if self.transforms is None:
            self.transforms = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(256),
                                transforms.ToTensor()])

    def __getitem__(self, idx):
        images = self.transforms(Image.open(
            os.path.join(self.data_dir, str(self.images[idx]))))
        text = open(os.path.join(self.data_dir, str(self.captions[idx])), 'r').read()
        texts = self.tokenize([text])[0]
        return images, texts


# TODO hardcoded image key
class ImageDataset(CustomDataset):
    """
    Dataset to load images from image-caption mapping file
    """
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = Image.open(
            os.path.join(self.data_dir, str(self.images[idx])))
        return {"image": self.transforms(images).permute(1, 2, 0)}


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

        # self.train_transform = _transform(image_resolution, True)
        # self.valid_transform = _transform(image_resolution, False)
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
        self.setup_fn = CustomDataset if setup_fn is None else setup_fn
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

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)


def get_clip_data(args, preprocess_fns):
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

# NOTE dalle tokenizer arg is part of model config
# model.tokenizer should be built first, then tokenize here is model.tokenizer.encode
# TODO shuffle arg for dalle dataloader
# TODO need to figure out how to handle transforms properly
def get_dalle_data(args, preprocess_fns, tokenize):
    preprocess_train, preprocess_val = preprocess_fns

    def setup_fn(transform, input_filename):
        return CustomDataset(input_filename, transform, args.img_key,
                             args.caption_key, tokenize, args.data_dir, args.csv_separator)

    return CustomDataModule(args.train_batch_size, args.valid_batch_size,
                            args.train_filename, args.valid_filename,
                            train_setup_fn=setup_fn(preprocess_train),
                            valid_setup_fn=setup_fn(preprocess_val), pin_memory=True)

def get_data(args, preprocess_fns, model_type, tokenize=None):
    if model_type == 'clip':
        return get_clip_data(args, preprocess_fns)
    elif model_type == 'dalle':
        assert tokenize is not None, "dalle tokenizer is part of model config"
        return get_dalle_data(args, preprocess_fns, tokenize)
    elif model_type == 'glide':
        raise NotImplementedError
    else:
        print("either get_data method is not implemented or you shouldn't be using it")
        raise NotImplementedError
