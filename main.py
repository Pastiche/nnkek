from nnkek.encoders import Autoencoder, get_dummy_batch
from nnkek.validation import TopKComparator, BootsTrapper, print_confidence_interval


if __name__ == '__main__':
    encoder = Autoencoder()

    raw = get_dummy_batch(16)
    encoded = encoder.encode(raw)

    comparator = TopKComparator(raw, encoded)

    bootstrapper = BootsTrapper(comparator)

    evaluation = bootstrapper.run()

    print_confidence_interval(evaluation, 100)