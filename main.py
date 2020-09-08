import cv2
from PIL import Image

from nnkek.encoders import Autoencoder, get_dummy_batch
from nnkek.validation import EncoderValidator
import math


if __name__ == '__main__':
    print(binary)

# Bootstraper():
    # get_random_bootstrap(data) - see Zhenya's case
# bootstrap_val(k=5) -> bootstrap of intersection of topk images before and after encoding
    # indices = get_topk_originals(index, k)
    # indices = get_topk_encoded(index, k)
    # intersection = intersect(indices, indices)
    # plot_topk_originals(index, k)
    # plot_topk_encoded(index, k)
    # plot_topk_comparison(index, k)
    # returns what? = bootstrap(k)