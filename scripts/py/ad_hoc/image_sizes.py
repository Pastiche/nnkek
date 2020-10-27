from PIL import Image

from nnkek.utils.path import get_img_paths

img_paths = get_img_paths("/data/shared/dataset/aer_images_ru")
print(len(img_paths))

small = [x for x in img_paths if Image.open(x).size[0] < 301 or Image.open(x).size[1] < 301]
print(len(small))
