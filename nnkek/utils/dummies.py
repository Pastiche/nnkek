import os

import numpy as np
import pandas as pd
import torch

DUMMY_IMG_DIR = "data/dummy/img"
DUMMY_ARRAY_DIR = "data/dummy/txt"

im_names = np.array(
    [
        "997722733_0cb5439472.jpg",
        "2473293833_78820d2eaa.jpg",
        "3104690333_4314d979de.jpg",
        "3522000960_47415c3890.jpg",
        "989851184_9ef368e520.jpg",
    ]
)

im_ids = np.array([x.split(".")[0] for x in im_names])
im_paths = np.array([os.path.join(DUMMY_IMG_DIR, x) for x in im_names])

arr_names = np.array(["a0.txt", "a1.txt", "a2.txt", "a3.txt", "a4.txt"])

arr_paths = np.array([os.path.join(DUMMY_ARRAY_DIR, x) for x in arr_names])

# these won't work since the root dir is not suitable
# ims = np.array([pil_load_image(x) for x in im_paths])
# arrays = np.array([np.loadtxt(x) for x in arr_paths])


def get_dummy_tensor(batch_size=16):
    return torch.FloatTensor(batch_size, 2048).uniform_(-10, 10)


def get_dummy_df():
    return pd.DataFrame({"A": ["lupa", "pupa", "lol", "kek"], "B": [1, 2, 5, 4]})
