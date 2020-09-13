import torch
from nnkek import plotters
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.utils import resample


class Displayer:
    def __init__(self, images, comparator):
        self.images = images
        self.comparator = comparator

    @staticmethod
    def decorate_ax(ax):
        ax.set_xticks([], minor=True)
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=True)
        ax.set_yticks([], minor=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    @staticmethod
    def decorate_anchor(ax):
        ax.set_title('Anchor', fontsize=14, color='red')
        ax.patch.set_linewidth('10')
        ax.patch.set_edgecolor('red')

    @staticmethod
    def decorate_match(ax):
        ax.patch.set_linewidth('10')
        ax.patch.set_edgecolor('yellow')

    def plot_closest_img_compare_row(self, indices, intersection):
        fig, ax = plotters.im_grid(self.images[indices],
                                   nrows=1,
                                   ncols=len(indices),
                                   figsize=(16, 4))

        for i, axi in enumerate(ax):
            self.decorate_ax(axi)
            if i == 0:
                self.decorate_anchor(axi)
            elif indices[i] in intersection:
                self.decorate_match(axi)
            else:
                axi.axis('off')

        fig.tight_layout()

    def plot_random_comparison(self):
        raw_topk_indices, encoded_topk_indices, intersect = self.comparator.take_random()
        intersect = torch.Tensor(intersect)

        self.plot_closest_img_compare_row(raw_topk_indices, intersect)
        self.plot_closest_img_compare_row(encoded_topk_indices, intersect)

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


class TopKComparator:
    def __init__(self,
                 raw: torch.Tensor,
                 encoded: torch.Tensor,
                 k=5,
                 largest=False):
        self.raw = TopK(raw, k, largest)
        self.encoded = TopK(encoded, k, largest)
        self.intersections = np.array([np.intersect1d(r, e)
                                       for r, e in zip(self.raw, self.encoded)])

    def __getitem__(self, i):
        return self.raw[i], self.encoded[i], self.intersections[i]

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


def bootstrap_result(true_labels, predicted_labels, amount=10000,
                     verbose_amount=500, label_names=None):
    bootstrap_results = []
    for i in range(amount):
        bootstrapped_true_labels, bootstrapped_predicted_labels = zip(
            *resample(list(zip(true_labels, predicted_labels)))
        )
        bootstrap_results.append(metrics.classification_report(
            list(bootstrapped_true_labels),
            list(bootstrapped_predicted_labels),
            output_dict=True,
            target_names=label_names,
            zero_division=0.0
        ))
        if verbose_amount is not None and (i + 1) % verbose_amount == 0:
            print(f"Processed {i + 1} samples")
    return bootstrap_results


class BootsTrapper:
    def __init__(self, comparator: TopKComparator):
        self.comparator = comparator

    def run(self, amount=10000, verbose_amount=500):
        bootstraps = []

        for i in range(amount):
            intersections = resample(self.comparator.intersections)
            # since anchor itself is in the intersection of topks,
            # subtract 1 to mitigate the bias
            # TODO: убрать это, а вместо этого сделать в компараторе параметр
            # include_anchor
            bootstraps.append(np.mean([len(x) - 1 for x in intersections]))

            if verbose_amount is not None and (i + 1) % verbose_amount == 0:
                print(f"Processed {i + 1} samples")

        return bootstraps


def draw_distribution(distr, use_0_to_1_range=False, bins=100, title=None):
    plt.hist(distr, bins=bins, range=(0, 1) if use_0_to_1_range else None)
    plt.title(title)
    plt.show()


def print_confidence_interval(distr, percentage=95):
    percentile_left = (100 - percentage) / 2
    percentile_right = 100 - percentile_left

    print(f"Confidence interval:")
    print(percentile_left, percentile_right)

    percentile_left_value = round(np.percentile(distr, percentile_left), 3)
    percentile_right_value = round(np.percentile(distr, percentile_right), 3)
    percentile_interval = percentile_right_value - percentile_left_value

    print(f'interval length: {round(percentile_interval, 3)}')

    print('[{}, {}]'.format(round(np.percentile(distr, percentile_left), 3),
                            round(np.percentile(distr, percentile_right), 3)))


def distribution_report(distr,
                        percentage=95,
                        use_0_to_1_range=False,
                        bins=100,
                        title=None):
    draw_distribution(distr, use_0_to_1_range, bins=bins, title=title)
    print_confidence_interval(distr, percentage)
    print()


def display_bootstrap_report(boorstraper, percentage=95, use_0_to_1_range=False,
                             amount=10000, verbose_amount=500, bins=100):
    bootstrap = boorstraper.run(amount=amount, verbose_amount=verbose_amount)

    distribution_report(bootstrap,
                        percentage=percentage,
                        use_0_to_1_range=use_0_to_1_range,
                        bins=bins)
