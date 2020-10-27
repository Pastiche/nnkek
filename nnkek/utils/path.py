import os.path as osp
from typing import Sequence

import numpy as np
from tools.util import file_utils as futils


def map_img_paths(image_folder: str, img_names: Sequence[str] = None, none_if_absent=True) -> dict:
    """Given files root folder and their unique names returns mapping to corresponding paths from the given root"""
    all_paths = {osp.basename(x): x for x in futils.path_get_file_list(image_folder, ["image"])}

    if img_names is None:
        return all_paths

    name2path = {}
    for x in img_names:
        img_path = all_paths.get(x)
        if img_path or none_if_absent:
            name2path[x] = img_path

    return name2path


def get_img_paths(image_folder: str, img_names: Sequence[str] = None, preserve_shape=True) -> np.array:
    """Given files root folder and their unique names returns corresponding paths from the given root"""
    name2path = map_img_paths(image_folder, img_names, none_if_absent=preserve_shape)

    paths = []

    if img_names is None:
        return np.array([x for x in name2path.values()])

    for img_name in img_names:
        img_path = name2path.get(img_name)
        if img_path or preserve_shape:
            paths.append(img_path)

    return np.array(paths)
