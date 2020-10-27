# -*- coding: utf-8 -*-
"""Fast multithreaded image deduplicator
"""
import argparse
import glob
import multiprocessing
import os
import time
from collections import defaultdict
import pandas as pd

import cv2
import imagehash
from PIL import Image


def calc_hash(image):
    h = imagehash.phash(Image.fromarray(image), hash_size=16)
    return h


def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        brief_hash = calc_hash(image)
    except:
        print("Invalid image: {}".format(image_path))
        return None
    return image_path, brief_hash


parser = argparse.ArgumentParser(description="Deduplicator")
parser.add_argument("--num-threads", type=int, default=8)
parser.add_argument("--delete", type=bool, default=True)
args = parser.parse_args()


phash = []
img = []

for dir in sorted(glob.glob("/data/shared/dataset/aer_images_glo/image/*")):
    print(dir)
    image_list = glob.glob(os.path.join(dir, "**/*"), recursive=True)
    image_list = [os.path.abspath(p) for p in image_list]

    print('Number of images in dir "{}" = {}'.format("", len(image_list)))

    print("Start processing with {} threads".format(args.num_threads))
    pool = multiprocessing.Pool(args.num_threads)
    ts = time.time()
    index = defaultdict(list)
    result = pool.imap(process_image, image_list, chunksize=500)

    for i, hashes in enumerate(result):
        if hashes is not None:
            img.append(hashes[0])
            phash.append(hashes[1])
        if i % 5000 == 0:
            print("processed {}/{}".format(i, len(image_list)))

    pool.close()
    pool.join()

    index_build_time = time.time() - ts
    print("index build time: {}".format(index_build_time))
    print(f"processed: {len(phash)}")

df = pd.DataFrame.from_dict({"item_image_name": img, "phash": phash})
df.to_csv("tables/phash_glo.csv", index=False)
