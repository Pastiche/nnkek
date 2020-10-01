import os
from typing import NewType
from typing import Tuple, Callable, Iterable

import numpy as np
import tensorflow as tf
import torch
from albumentations import BasicTransform
from tools.util import file_utils
from torchvision.transforms import transforms

from nnkek.encoders import get_device


class TorchImgTransfomr(BasicTransform):
    def get_params_dependent_on_targets(self, params):
        pass

    def __init__(self):
        super().__init__()
        self.vectorizer = TorchImgVectorizer()

    def apply(self, img, **params):
        return self.vectorizer(img)

    def get_params(self):
        return {}

    @property
    def targets(self):
        return None


class TorchImgVectorizer:
    def __init__(self, model=None, preproc=None):
        self.model = model or self.get_default_model()
        self.preproc = preproc or self.get_default_transform()
        self.model.to(get_device())

    def __call__(self, batch: torch.Tensor):
        return self.transform(batch)

    @staticmethod
    def get_default_model():
        return torch.hub.load("pytorch/vision:v0.6.0", "inception_v3", pretrained=True)

    @staticmethod
    def get_default_transform():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def transform(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch = batch.to(get_device())
            batch = self.preproc(batch)
            return self.model(batch)


LoadImgCallback = NewType("ImgLoadCallback", Callable[[tf.Tensor], Tuple[np.array, tf.Tensor]])


def load_img(im_path: tf.Tensor) -> tf.Tensor:
    im = tf.io.read_file(im_path)
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.resize(im, (299, 299))
    im = tf.cast(im, tf.float32)
    im = tf.keras.applications.inception_v3.preprocess_input(im)
    return im


class TfIndexableDataset(torch.utils.data.Dataset):
    def __init__(self, im_paths, load_img_callback=None, n_cpu: int = tf.data.experimental.AUTOTUNE):

        self.n_cpu = n_cpu
        self.im_paths = np.array(im_paths)
        self.load_img = load_img_callback or load_img

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.load_img(self.im_paths[index])
        # workaround: one_shot (single batch) dataset and then iterate once
        dataset = tf.data.Dataset.from_tensor_slices(self.im_paths[index])
        dataset = dataset.map(self.load_img, num_parallel_calls=self.n_cpu).batch(len(dataset))

        return next(iter(dataset))


class ImageVectorizer:
    def __init__(self, model=None, load_img_callback: LoadImgCallback = None):
        """
        Можно считать этот векторайзер упраздненным. Во-первых, он создан
        как бы для перелопачивания всех данных сразу, а не загрузкой батчами,
        во-вторых, он на сраном тензорфлоу. Держу его лишь ради ноутбука, где
        валидируется энкодер. Но можно просто сейчас его заменить на новый.

        :param model: callable model which takes a (supposedly 4D) array
        representing batch of images and returns a batch of embeddings.
        :param load_img_callback: a function for image loading and preprocessing

        Default is InceptionV3 model with its corresponding preprocessing
        """

        self.model = model or self.__get_default_model()
        self.load_img = load_img_callback or self.__load_img

    def __call__(self, img_batch: tf.Tensor):
        return self.model(img_batch)

    @staticmethod
    def __get_default_model():
        model = tf.keras.applications.InceptionV3()
        return tf.keras.Model(model.input, model.layers[-1].input)

    @staticmethod
    def __load_img(img_path: tf.Tensor) -> Tuple[np.array, tf.Tensor]:
        return load_img(img_path), img_path

    def process_and_persist(self, img_paths: Iterable[str], n_cpu: int = tf.data.experimental.AUTOTUNE, batch_size=16):
        for features_batch, path_batch in self.process(img_paths, n_cpu, batch_size):
            self.persist_img_features_batch(features_batch, path_batch)

    def process(self, img_paths: Iterable[str], n_cpu: int = tf.data.experimental.AUTOTUNE, batch_size=16):
        """Returns a generator, will yields tuple of
        (image features batch, corresponding image paths batch)
        """

        # Construct a lazy loader
        img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
        img_dataset = img_dataset.map(self.load_img, num_parallel_calls=n_cpu)
        img_dataset = img_dataset.batch(batch_size)

        # extract image features
        for img_batch, path_batch in img_dataset:
            yield self.model(img_batch), path_batch

    def persist_img_features_batch(self, batch_features, path_batch):
        for features, img_path in zip(batch_features, path_batch):
            self.persist_img_features(features, img_path)

    @staticmethod
    def persist_img_features(img_features, img_path):
        # decode path and remove extension
        img_path = img_path.numpy().decode("utf-8").split(".")[0]
        # np.save adds '.npy' automatically
        np.save(img_path, img_features.numpy())


def from_dir(vectorizer, folder, rewrite=False):
    """
    :vectorizer: model for extracting features from images
    :img_paths: iterable of image paths with extentions
    """

    img_paths = file_utils.path_get_file_list(folder, ["img"])[0]
    from_paths(vectorizer, img_paths, rewrite)


def from_paths(vectorizer, img_paths, rewrite=False):
    """
    :vectorizer: model for extracting features from images
    :img_paths: iterable of image paths with extentions
    """

    if not rewrite:
        img_paths = [x for x in img_paths if not os.path.exists(x.split(".")[0] + ".npy")]

    unqiue_images = set(img_paths)
    print("{} unique images to process..".format(len(unqiue_images)))

    if unqiue_images:
        vectorizer.process_and_persist(list(unqiue_images))
