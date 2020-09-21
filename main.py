from torch.utils.data import DataLoader

from nnkek import dummies, embeddings
from nnkek.augmentation import get_default_transform
from nnkek.embeddings import TfIndexableDataset, TorchImgVectorizer
from nnkek.encoders import Autoencoder, get_dummy_batch
from nnkek.imagers import imshow
from nnkek.datasets import ImDataset, ImAugDataset, ArrayDataset
from nnkek.plotters import im_grid
from nnkek.validation import TopKComparator, BootsTrapper, \
    print_confidence_interval

import matplotlib.pyplot as plt


def test_aug():
    # запомните твари, берете батч и каждую картинку аугментите с 50% вероятностью
    transform = get_default_transform()

    dataset = ImAugDataset(dummies.im_paths, transform)

    print('multiple indexing:')
    pics = dataset[[1, 3]]
    print(pics.shape)

    print('single indexing:')
    pic = dataset[1]
    print(pic.shape)

    print('batched, shuffled dataloader:')
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch.shape)


def test_array_dataset():
    dataset = ArrayDataset(dummies.arr_paths)

    print('multiple indexing:')
    arrays = dataset[[1, 3]]
    print(arrays)

    print('single indexing:')
    array = dataset[1]
    print(array)

    print('batched, shuffled dataloader:')
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch)


def test_safe_index():
    orig_im_dataset = ImDataset(dummies.im_paths)
    for i in range(len(dummies.im_paths)):
        imshow(orig_im_dataset[i])
        plt.show()

    ims = orig_im_dataset[[1, 3, 4]]
    im_grid(ims, nrows=1, ncols=3)


def test_vectorizer():
    vectorizer = embeddings.ImageVectorizer()
    for batch in vectorizer.process(list(dummies.im_paths)):
        print(batch)


def test_tf_dataset():
    dataset = TfIndexableDataset(list(dummies.im_paths))
    pic = dataset[4]
    print(pic)

    pic = dataset[4]
    print(pic)

    pics = dataset[[1, 4]]  # indexing [1, 4] would pass a tuple!
    print(pics)


def test_encoder_bootstrap():
    encoder = Autoencoder()
    raw = get_dummy_batch(16)
    encoded = encoder.encode(raw)

    comparator = TopKComparator(raw, encoded)
    bootstrapper = BootsTrapper(comparator)
    evaluation = bootstrapper.run()

    print_confidence_interval(evaluation, 100)


def test_vectorizer_torch():
    dataset = ImAugDataset(dummies.im_paths)
    vectorizer = TorchImgVectorizer()
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print(batch.shape)
        processed = vectorizer.transform(batch)
        print(processed)


if __name__ == '__main__':
    # img = cv2.imread('data/Flicker8k/Flicker8k_Dataset/667626_18933d713e.jpg')
    # print(img.shape)
    # test_aug()
    # test_vectorizer()
    # test_tf_dataset()
    # test_safe_index()
    # test_array_dataset()
    test_vectorizer_torch()
