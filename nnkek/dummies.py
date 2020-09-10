import os
from tools.util.img.io_utils import pil_load_image

FLICKER_DATA_DIR = 'data/Flicker8k/Flicker8k_Dataset'

im_names = [
    '997722733_0cb5439472.jpg',
    '2473293833_78820d2eaa.jpg',
    '3104690333_4314d979de.jpg',
    '3522000960_47415c3890.jpg',
    '989851184_9ef368e520.jpg'
]

im_paths = map(lambda x: os.path.join(FLICKER_DATA_DIR, x), im_names)

ims = map(lambda x: pil_load_image, im_paths)