import statistics
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nnkek import imagers


def im_grid(images, nrows=2, ncols=5, figsize=(6, 8), imgsize=(128, 128)):
    # это негибко. Лучше, чтобы кол-во картинок было неожиданным
    assert len(images) == nrows * ncols

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if nrows * ncols == 1:
        ax = [ax]

    for i, axi in enumerate(ax):
        img = images[i]
        imagers.imshow(img, axi, imgsize)

    return fig, ax


def counterplot_with_caps(data):
    sns.set(rc={"figure.figsize": (14, 7)})
    counter = Counter(data)
    c = sns.countplot(x=data)

    c.axes.set_title(f"Kek", fontsize=16)
    c.set_xlabel("Median: {}, Mean: {:2.1f}".format(statistics.median(data), statistics.mean(data)), fontsize=16)
    c.set_ylabel("", fontsize=16)
    c.tick_params(labelsize=16)

    for i, v in counter.items():
        ratio = "{:2.1f}%".format(v / np.sum(list(counter.values())) * 100)
        c.text(i, v, ratio, color="gray", fontsize=14, ha="center")

    plt.show()
