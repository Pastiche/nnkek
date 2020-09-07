import cv2
import os
import urllib
import urllib.request
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

tips = """

CV2
img = cv2.imread(img_path)
<class 'numpy.ndarray'>
(300, 300, 3)
[[[255 255 255]
  [255 255 255]
  [255 255 255]
  ...
cv2.resize(img, size)
cv2.imwrite(img_path, img)

KERAS (PIL)
keras.preprocessing.image.load_img()
<class 'PIL.JpegImagePlugin.JpegImageFile'>
(300, 300)
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x300 at 0x7F0574F768D0>

np.asarray(pil_img) ->
array([[[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],

load_img(images[i], target_size=(128, 128))


plt.imshow(img) works for both!!!


"""

temp_dir = "img_cache"
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)


def show_img(img_path):
    img = cv2.imread(img_path)
    plt.imshow(img)


def download_and_resize_img(img_name, target_dir=temp_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if len(img_name) == 0:
        return "empty.jpg"
    img_url = "https://ae01.alicdn.com/kf/{}".format(img_name)
    img_path = os.path.join(temp_dir, img_name)
    if os.path.exists(img_path):
        return img_path
    urllib.request.urlretrieve(img_url, img_path)
    img = cv2.imread(img_path)
    # print(img.shape)
    scale = 300 / max(img.shape[0:1])
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    cv2.imwrite(img_path, img)
    return img_path


def show_image_row(images, decorate_ax=None, decorate_fig=None, caption=None):
    # https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645

    fig, axis = plt.subplots(nrows=1, ncols=len(images), figsize=(20, 4))  # row number, row lenght, column number

    if decorate_fig:
        decorate_fig(fig)

    for i, ax in enumerate(axis.flat):
        img = images[i]
        if isinstance(img, str):
            # img = load_img(images[i], target_size=(128, 128))
            img = cv2.imread(img)
        img = cv2.resize(img, (128, 128))

        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decorate_ax:
            decorate_ax(img, ax, str(images[i]), i, len(images))

    plt.tight_layout(True)

# def decorate_ax(img, ax, img_name=None, img_number=None, row_length):
#     pass

# def decorate_fig(fig):
#     pass
