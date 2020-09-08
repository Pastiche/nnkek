import torch
import numpy as np
import matplotlib.pyplot as plt

from nnkek import plots


def decorate_ax(ax, img_number, topk_indices1, topk_indices2):
    ax.set_xticks([], minor=True)
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=True)
    ax.set_yticks([], minor=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if img_number == 0:
        ax.set_title('Anchor', fontsize=20, color='red')
        ax.patch.set_linewidth('10')
        ax.patch.set_edgecolor('red')
        return

    elif topk_indices1[img_number] in topk_indices2:
        ax.patch.set_linewidth('10')
        ax.patch.set_edgecolor('yellow')

    else:
        ax.axis('off')


def plot_closest_img_compare_row(images, this_indices, other_indices):
    fig, ax = plots.img_grid(images,
                             nrows=1,
                             ncols=len(images),
                             figsize=(16, 10))

    for i, axi in enumerate(ax):
        decorate_ax(axi, i, this_indices, other_indices)
    plt.tight_layout()


def plot_random_comparison(validator, images):
    raw_topk_indices, encoded_topk_indices = validator.take_random()

    raw_topk = images[raw_topk_indices]
    encoded_topk = images[encoded_topk_indices]

    plot_closest_img_compare_row(raw_topk,
                                 raw_topk_indices,
                                 encoded_topk_indices)

    plot_closest_img_compare_row(encoded_topk,
                                 encoded_topk_indices,
                                 raw_topk_indices)

    plt.show()


class TopK:
    """
    given a set of vectors computes top k with largest/smallest Euclidian norm.
    Note that for largest=False the closest one will always be the target vector
    itself.
    """
    def __init__(self, vectors: torch.Tensor, k=5, largest=False):
        distances = torch.cdist(vectors, vectors)
        self.distances, self.indices = torch.topk(distances, k, largest=largest)

    def __getitem__(self, i: int):
        return self.indices[i]

    @property
    def shape(self):
        return self.indices.shape

    def __len__(self):
        return self.shape[0]


class EncoderValidator:
    def __init__(self,
                 raw: torch.Tensor,
                 encoded: torch.Tensor,
                 k=5,
                 largest=False):

        self.raw = TopK(raw, k, largest)
        self.encoded = TopK(encoded, k, largest)

    def __getitem__(self, i: int):
        return self.raw[i], self.encoded[i]

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def take_random(self):
        return self[np.random.randint(self.__len__())]

    @property
    def shape(self):
        return self.raw.shape

    def __len__(self):
        return self.shape[0]
