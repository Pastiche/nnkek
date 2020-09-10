import os
from typing import Tuple

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tools.util.img.common_utils import size_yx
from tools.util.img.io_utils import pil_download_image_by_url
from tools.util.img.proc_utils import resize_image


def is_valid_image(img):
    try:
        if isinstance(img, str):
            img = Image.open(img)
        img.verify()
    except (IOError, SyntaxError) as e:
        return False
    return True


def imshow(img, canvas=None, imgsize: Tuple = None, no_axis=True):

    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(img)

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv has BGR by default

    if isinstance(img, np.ndarray) and imgsize:
        img = cv2.resize(img, imgsize)

    if isinstance(img, PIL.Image.Image) and imgsize:
        img.resize(imgsize)

    canvas = canvas or plt
    if no_axis:
        plt.axis('off')

    canvas.imshow(img)


class ImgLoader():
    def __init__(self, folder=None):
        self.folder = folder

    def download_to_folder(self, image_aeid, file_name,
                           check_if_exists=True, verbose=False,
                           min_size=256, max_size=1024, return_path=False):
        assert file_name is not None
        assert image_aeid is not None
        assert self.folder is not None

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        return self.download(image_aeid=image_aeid,
                             file_name=os.path.join(self.folder, file_name),
                             check_if_exists=check_if_exists,
                             verbose=verbose,
                             min_size=min_size,
                             max_size=max_size,
                             return_name=return_path)

    def download(self, image_aeid, file_name=None,
                 check_if_exists=True, verbose=False,
                 min_size=256, max_size=1024, return_name=False):
        # loads and resizes an image; persists if file_name is provided,
        # else returns PIL.Image.Image wrapper around image loaded into RAM

        # skips if height of width < min_size
        # resizes to weight*height/max_size^2 if any height or width > max_size
        # resize preserves image initial weight/height ratio

        if check_if_exists and file_name and os.path.exists(file_name):
            if verbose:
                print(f'Skip existing: {file_name}')
            return file_name if return_name else None

        # this call will only load the image into RAM
        img = pil_download_image_by_url(image_aeid, verbose=verbose)

        if not img:
            # the error is logged in the called function
            return

        if not is_valid_image(img):
            if verbose:
                print(f'Invalid image: {image_aeid}')
            return

        # skip small
        if min_size and (img.width < min_size or img.height < min_size):
            if verbose:
                print('Too small, skipped: {} ({}x{})'.format(
                    image_aeid, img.img_width, img.height
                ))

        # resize big
        if max_size:
            img = resize_(img, max_size)

        # store
        if file_name:
            with open(file_name, 'wb') as f:
                img.save(f)

        return file_name if return_name and return_name else img


def resize_(img, size):
    decrease_ratio = np.sqrt(np.prod(size_yx(img)) / (size ** 2))
    if decrease_ratio > 1:
        img, _ = resize_image(img, interpolation='bicubic',
                              magnify_ratio=1 / decrease_ratio)
    return img
