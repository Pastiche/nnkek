import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np


def img_grid(images, nrows=2, ncols=5, figsize=(6, 8), imgsize=(128, 128)):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        img = images[i]

        if isinstance(img, str):
            img = cv2.imread(img)

        if isinstance(img, np.ndarray):
            img = cv2.resize(img, imgsize)

        if isinstance(img, PIL.Image.Image):
            img.resize(imgsize)

        axi.imshow(img)

    return fig, ax
