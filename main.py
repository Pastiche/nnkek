from torch.utils.data import DataLoader

from nnkek import dummies
from nnkek.augmentation import get_default_transform, AlbumentationImageDataset
from nnkek.encoders import Autoencoder, get_dummy_batch
from nnkek.validation import TopKComparator, BootsTrapper, \
    print_confidence_interval


def test_aug():
    transform = get_default_transform()

    dataset = AlbumentationImageDataset(im_paths=list(dummies.im_paths),
                                        transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    for im_path, aug_im in dataloader:
        print(im_path)
        print(aug_im.shape)
        break


def test_encoder_bootstrap():
    encoder = Autoencoder()
    raw = get_dummy_batch(16)
    encoded = encoder.encode(raw)

    comparator = TopKComparator(raw, encoded)
    bootstrapper = BootsTrapper(comparator)
    evaluation = bootstrapper.run()

    print_confidence_interval(evaluation, 100)


if __name__ == '__main__':
    # img = cv2.imread('data/Flicker8k/Flicker8k_Dataset/667626_18933d713e.jpg')
    # print(img.shape)
    test_aug()
