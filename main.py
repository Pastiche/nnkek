import cv2

from nnkek.encoders import Autoencoder, get_dummy_batch
from nnkek.validation import TopKComparator, BootsTrapper, print_confidence_interval


if __name__ == '__main__':
    img = cv2.imread('data/Flicker8k/Flicker8k_Dataset/667626_18933d713e.jpg')
    print(img.shape)

    # encoder = Autoencoder()
    #
    # raw = get_dummy_batch(16)
    # encoded = encoder.encode(raw)
    #
    # comparator = TopKComparator(raw, encoded)
    #
    # bootstrapper = BootsTrapper(comparator)
    #
    # evaluation = bootstrapper.run()
    #
    # print_confidence_interval(evaluation, 100)