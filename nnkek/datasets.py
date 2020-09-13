from typing import Tuple, Callable

import torch
from tools.util.img.io_utils import cv2_load_image, pil_load_image
import numpy as np
from torch.utils.data import Dataset
from collections.abc import Iterable


class AbstractMapper(Dataset):
    """This maps entities, e.g. item_id to one of it's real photos or titles"""
    # do we really need this? We might build it later if needed
    # might be useful when we have multiple images of the same item
    pass


class SafeIndexer(Dataset):
    """Provides iterable indexing for those datasets which don't have it
    themselves"""

    def get_one(self, i):
        raise NotImplementedError

    def get_one_or_more(self, index):
        if isinstance(index, int):
            return self.get_one(index)
        elif isinstance(index, Tuple):
            raise IndexError('tuple indexing is not allowed')
        elif isinstance(index, Iterable):
            return np.array([self.get_one(i) for i in index])
        else:
            raise IndexError(f'{type(index)} is not supported index key type')


class ImDataset(SafeIndexer):
    def __init__(self, im_paths: np.ndarray):
        self.im_paths = im_paths

    def __getitem__(self, i):
        return super(ImDataset, self).get_one_or_more(i)

    def __len__(self):
        return len(self.im_paths)

    def get_one(self, i) -> np.ndarray:
        return cv2_load_image(self.im_paths[i])


class ImAugDataset(ImDataset):
    def __init__(self, im_paths, transform=None):
        super(ImAugDataset, self).__init__(im_paths)
        self.aug = transform

    def __getitem__(self, i) -> np.ndarray:
        return super(ImAugDataset, self).get_one_or_more(i)

    def get_one(self, i) -> np.ndarray:
        im = super(ImAugDataset, self).get_one(i)
        return self.aug(image=im)['image']  # not flexible, should be changed


class ArrayDataset(SafeIndexer):
    def __init__(self, array_paths):
        self.array_paths = array_paths

    def __getitem__(self, i) -> np.ndarray:
        return super(ArrayDataset, self).get_one_or_more(i)

    def __len__(self):
        return len(self.array_paths)

    def get_one(self, i) -> np.ndarray:
        return np.loadtxt(self.array_paths[i])
