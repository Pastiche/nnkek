import albumentations as A
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from nnkek import dummies


def get_default_transform():
    return A.Compose([
        A.Resize(300, 300),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45,
                           p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.ToFloat(always_apply=True, p=1.0)
    ])


class AlbumentationImageDataset(Dataset):
    def __init__(self, im_paths, transform):
        self.ims = im_paths
        self.aug = transform

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        image = plt.imread(self.ims[i])
        if image.dtype == np.float32:
            image = image * 255
            image = image.astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))['image']

        return self.ims[i], torch.tensor(image, dtype=torch.float)
