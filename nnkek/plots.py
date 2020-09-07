import cv2
import matplotlib.pyplot as plt


def img_grid(images, nrows=2, ncols=5, figsize=(6, 8)):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        img = images[i]

        if isinstance(img, str):
            img = cv2.imread(img)

        axi.imshow(img, alpha=0.25)

    return fig, ax
