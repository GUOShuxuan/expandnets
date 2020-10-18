"""
Implement the generic dataset by ImageFolder.

Please define the DATASET_ROOT first to your own data path.

"""
import logging
import os
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset as pt_Dataset
import torchvision.datasets as tv_datasets
from torchvision.datasets.folder import make_dataset, find_classes, IMG_EXTENSIONS, default_loader


DATASET_ROOT = "<PATH-To-IMAGENET-DATASET>"


class CompositeDatasets():
    """
    This is a abstract dataset and is a template for train_mobilenets.py, infer.py and test.py.

    """

    train_dataset = None
    valid_dataset = None
    test_dataset = None

    def __init__(self, root, cfg, dataset_fn, train_configs, valid_configs, test_configs):
        if not 'root' in train_configs.keys():
            train_configs['root'] = root

        if not 'root' in valid_configs.keys():
            valid_configs['root'] = root

        if not 'root' in test_configs.keys():
            test_configs['root'] = root

        self.train_dataset = dataset_fn(**train_configs)
        self.valid_dataset = dataset_fn( **valid_configs)
        self.test_dataset = dataset_fn(**test_configs)
        self.cfg = cfg


class DatasetFolderv2(tv_datasets.DatasetFolder):

    meta_file = 'preprocessed.pkl'

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.transform = transform
        self.target_transform = target_transform
        samples = self._extract_dataset()

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

    def _check_Integrity(self):
        return os.path.exists(os.path.join(self.root, self.meta_file))

    def _extract_dataset(self):
        if self._check_Integrity():
            logging.info(f"Load meta info into {self.meta_file}")
            self.classes, self.class_to_idx, self.samples = pickle.load(open(
                os.path.join(self.root, self.meta_file), 'rb'))
        else:
            self.classes, self.class_to_idx = find_classes(self.root)
            self.samples = make_dataset(self.root, self.class_to_idx, self.extensions)
            pickle.dump((self.classes, self.class_to_idx, self.samples),
                        open(os.path.join(self.root, self.meta_file), 'wb'))
            logging.info(f"Processed dataset meta info into {self.meta_file}")
        return self.samples


class ImageFolderv2(DatasetFolderv2):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolderv2, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


def classes_one_hot(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


def read_txt_without_endline(file_path):
    res = tuple(open(file_path))
    res = [id_.rstrip() for id_ in res]
    return res


# Data Loading functions {{{
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def prefetched_loader(loader, fp16):
    """
    Stream loader
    :param loader:
    :param fp16:
    :return:
    """
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
    if fp16:
        mean = mean.half()
        std = std.half()

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(async=True)
            next_target = next_target.cuda(async=True)
            if fp16:
                next_input = next_input.half()
            else:
                next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target


def get_loader(dataset, batch_size, split='train', workers=5, _worker_init_fn=None):

    print("=>  Start process train folder...")

    if split == 'train':
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler,
            collate_fn=fast_collate, drop_last=True)
        print("Finish process train loader!")

    elif split in ['valid', 'test']:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)
        print("Finish process validation loader! ")
    else:
        raise NotImplementedError("Test loader not supported")

    return loader

