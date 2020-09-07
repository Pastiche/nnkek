from nnkek.encoders import Autoencoder, get_dummy_batch
from nnkek.validation import EncoderValidator


if __name__ == '__main__':
    encoder = Autoencoder()
    raw = get_dummy_batch(16)
    encoded = encoder.encode(raw)
    # print(raw.shape)
    # print(encoded.shape)

    validator = EncoderValidator(raw, encoded, 5)

    # print(iterator)

    for kek, lol in validator:
        print(kek, lol)

