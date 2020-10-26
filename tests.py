import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from nnkek import embeddings
from nnkek.augmentation import get_default_transforms
from nnkek.datasets import ImDataset, ImAugDataset, ArrayDataset
from nnkek.embeddings import TfIndexableDataset, TorchImgVectorizer
from nnkek.encoders import Autoencoder
from nnkek.imagers import imshow
from nnkek.plotters import im_grid
from nnkek.utils import dummies
from nnkek.utils.dummies import get_dummy_tensor
from nnkek.utils.math import dist_batch_parallel, dist_batch
from nnkek.utils.path import map_img_paths, get_img_paths
from nnkek.validation import TopKComparator, BootsTrapper, print_confidence_interval
from scripts.py.clustering import build_disjoint_sets, cluster_rec


def test_load_aug_vect_enc():
    transforms = get_default_transforms()

    vectorizer = TorchImgVectorizer()

    dataset = ImAugDataset(dummies.im_paths, transforms)

    print("multiple indexing:")
    pics = dataset[[1, 3]]
    print(pics.shape)

    print("single indexing:")
    pic = dataset[1]
    print(pic.shape)

    print("batched, shuffled dataloader:")
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch.shape)


def test_aug():
    # запомните твари, берете батч и каждую картинку аугментите с 50% вероятностью
    transform = get_default_transforms()

    dataset = ImAugDataset(dummies.im_paths, transform)

    print("multiple indexing:")
    pics = dataset[[1, 3]]
    print(pics.shape)

    print("single indexing:")
    pic = dataset[1]
    print(pic.shape)

    print("batched, shuffled dataloader:")
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch.shape)


def test_array_dataset():
    dataset = ArrayDataset(dummies.arr_paths)

    print("multiple indexing:")
    arrays = dataset[[1, 3]]
    print(arrays)

    print("single indexing:")
    array = dataset[1]
    print(array)

    print("batched, shuffled dataloader:")
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
    raw = get_dummy_tensor(16)
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


def test_oss_dataset():
    pass
    # dataset = OSSDataset(endpoint, bucket, index_file)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=num_loaders,
    #     pin_memory=True)


def test_img_paths():
    mapping = map_img_paths(image_folder="data/dummy/img", img_names=["989851184_9ef368e520.jpg", "kek.jpg"])
    print(mapping)
    mapping = map_img_paths(
        image_folder="data/dummy/img", img_names=["989851184_9ef368e520.jpg", "kek.jpg"], none_if_absent=False
    )
    print(mapping)
    path_list = get_img_paths(image_folder="data/dummy/img", img_names=["989851184_9ef368e520.jpg", "kek.jpg"])
    print(path_list)
    path_list = get_img_paths(
        image_folder="data/dummy/img", img_names=["989851184_9ef368e520.jpg", "kek.jpg"], preserve_shape=False
    )
    print(path_list)


def test_cluster_rec(steps=3):
    df = pd.DataFrame(
        {"klaster": list("ABCBBDCDED"), "kektor": [np.random.randint(0, 10, 10) for _ in range(len("ABCDEFGHIJ"))]}
    )

    print(df)
    print(df.shape)
    res = cluster_rec(df, cluster_col="klaster", vectors_col="kektor", steps=steps, threshold=5.0, threshold_step=1.5)
    print(res)


def test_disjoint_sets():
    # sets: [{0, 4, 5}, {1, 2, 3}]
    neighborhoods = [[4, 5], [2], [1, 3], [2], [0], [0]]
    sets = build_disjoint_sets(neighborhoods)
    print(sets)  # [5, 3, 3, 3, 5, 5]


def test_cdist_batch():
    vectors_raw = [[random.randint(0, 10) for _ in range(4)] for _ in range(100)]
    vectors = np.asarray(vectors_raw)
    distances = dist_batch(vectors, batch_size=2)
    print(distances)
    print(distances.shape)


def test_cdist_batch_parallel():
    vectors_raw = [[random.randint(0, 10) for _ in range(4)] for _ in range(100)]
    vectors = np.asarray(vectors_raw)
    print(vectors.shape)
    distances = dist_batch_parallel(vectors)
    print(distances)
    print(distances.shape)
    print(len(distances))


def profile(f, **args):
    ts = time.time()
    res = f(**args)
    print(f"{f} {time.time() - ts}")
    return res


if __name__ == "__main__":
    pass
    # test_aug()
    # test_vectorizer()
    # test_tf_dataset()
    # test_safe_index()
    # test_array_dataset()
    # test_vectorizer_torch() # TODO: fix transform (takes PIL image, while I give torch.Tensor)
    # test_img_paths()
    # test_cdist_batch_parallel()
    # test_cdist_batch()
    # test_cdist_batch_parallel()
    # test_disjoint_sets()
    # test_cluster_rec()
