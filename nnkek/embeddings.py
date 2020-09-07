from typing import Tuple, Callable, Iterable

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from typing import NewType
from tools.util import file_utils
import os

LoadImgCallback = NewType(
    'ImgLoadCallback',
    Callable[[tf.Tensor], Tuple[np.array, tf.Tensor]]
)


class ImageVectorizer:
    def __init__(self, model=None, load_img: LoadImgCallback = None):
        """
        :param model: callable model which takes a (supposedly 4D) array
        representing batch of images and returns a batch of embeddings.
        :param load_img: a function for loading and preprocessing images

        Default is InceptionV3 model with its corresponding preprocessing
        """

        self.model = model or self.__get_default_model()
        self.load_img = load_img or self.__load_img

    def __call__(self, img_batch: tf.Tensor):
        return self.model(img_batch)

    @staticmethod
    def __get_default_model():
        model = tf.keras.applications.InceptionV3()
        return tf.keras.Model(model.input, model.layers[-1].input)

    @staticmethod
    def __load_img(img_path: tf.Tensor) -> Tuple[np.array, tf.Tensor]:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, img_path

    def process_and_persist(
            self,
            img_paths: Iterable[str],
            n_cpu: int = tf.data.experimental.AUTOTUNE,
            batch_size=16
    ):
        for features_batch, path_batch in self.process(
                img_paths, n_cpu, batch_size
        ):
            self.persist_img_features_batch(features_batch, path_batch)

    def process(
            self,
            img_paths: Iterable[str],
            n_cpu: int = tf.data.experimental.AUTOTUNE,
            batch_size=16
    ):
        """Returns a generator, will yields tuple of
           (image features batch, corresponding image paths batch)
        """

        # Construct a lazy loader
        img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
        img_dataset = img_dataset.map(self.load_img, num_parallel_calls=n_cpu)
        img_dataset = img_dataset.batch(batch_size)

        # extract image features and store as numpy
        for img_batch, path_batch in tqdm(img_dataset):
            yield self(img_batch), path_batch

    def persist_img_features_batch(self, batch_features, path_batch):
        for features, img_path in zip(batch_features, path_batch):
            self.persist_img_features(features, img_path)

    @staticmethod
    def persist_img_features(img_features, img_path):
        # decode path and remove extension
        img_path = img_path.numpy().decode("utf-8").split('.')[0]
        # np.save adds '.npy' automatically
        np.save(img_path, img_features.numpy())


def from_dir(vectorizer, folder, rewrite=False):
    """
    :vectorizer: model for extracting features from images
    :img_paths: iterable of image paths with extentions
    """

    img_paths = file_utils.path_get_file_list(folder, ['img'])[0]
    from_paths(vectorizer, img_paths, rewrite)


def from_paths(vectorizer, img_paths, rewrite=False):
    """
    :vectorizer: model for extracting features from images
    :img_paths: iterable of image paths with extentions
    """

    if not rewrite:
        img_paths = [x for x in img_paths
                     if not os.path.exists(x.split('.')[0] + '.np')]

    unqiue_images = set(img_paths)
    print('{} images, {} unique'.format(len(img_paths), len(unqiue_images)))

    if unqiue_images:
        vectorizer.process_and_persist(list(unqiue_images))
