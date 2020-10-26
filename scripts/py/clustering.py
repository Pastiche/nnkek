__author__ = "akaiashi"

import argparse
import multiprocessing as mp
import os.path as osp
import random
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from disjoint_set import DisjointSet
from nnkek import plotters, utils as kutils
from nnkek.utils.math import dist_batch_parallel, dist_batch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from timm.utils import *
from tqdm import tqdm
import logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_npz",
        type=str,
        default="/tmp/features.npz",
        help="path to image features npz with fields: sample_ids, embeddings",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="/tmp/clusters.csv",
        help="where to save the resulting csv",
    )

    parser.add_argument(
        "--phash_csv",
        type=str,
        default=None,
        help="path to csv with fields item_image_name, phash. If provided first round clustering will be performed by phash",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="how many iterations of clustering to perform",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="if two vectors have distance less than threshold, they will be assigned to the same cluster",
    )

    parser.add_argument(
        "--threshold_step",
        type=float,
        default=1.2,
        help="how much to multiply the threshold each iteration",
    )

    parser.add_argument("--num_threads", type=int, default=-1, help="if != 1 distances will be computed in parallel")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="If --num_threads == 1 batches are computed sequentially, in parallel otherwise."
        "If not provided, but --num_threads != 1, batch size will be set as len(data) // num_threads",
    )

    parser.add_argument("--pca_dim", type=int, default=None, help="number of pca components")

    parser.add_argument("--verbose", type=bool, default=False, help="logging_lvl = DEBUG if args.verbose else INFO")

    return parser.parse_args()


def build_disjoint_sets(neighbourhoods: Sequence[Sequence[int]]):
    """neighborhoods[i] is a neighbourhood of the element i which may or may not include the i-th element itself"""

    rec_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(rec_limit, len(neighbourhoods)))

    ds = DisjointSet()

    if logging.getLogger().level <= logging.DEBUG:
        neighbourhoods = tqdm(neighbourhoods)

    for element, neighbourhood in enumerate(neighbourhoods):
        for neighbour in neighbourhood:
            ds.union(element, neighbour)

    sys.setrecursionlimit(rec_limit)

    return [ds.find(x) for x in range(len(neighbourhoods))]


def get_distance_sets(vectors: Sequence[Sequence[float]], threshold=5.0, batch_size=None, n_jobs=-1):
    logging.debug("computing distances..")
    vectors = np.asarray(vectors)
    logging.debug(f"vectors mean: {vectors.mean()}, std: {vectors.std()}")

    if n_jobs == -1:
        n_jobs = mp.cpu_count() - 1

    if n_jobs != 1:
        if not batch_size:
            batch_size = vectors.shape[0] // n_jobs
        distances = dist_batch_parallel(vectors, batch_size=batch_size, n_jobs=n_jobs)
    else:
        distances = dist_batch(vectors, batch_size=batch_size) if batch_size else cdist(vectors, vectors)

    logging.debug(f"distances mean: {distances.mean()}, std: {distances.std()}")

    logging.debug("getting neighbours..")
    neighbourhoods = [np.argwhere(x < threshold).flatten() for x in distances]  # includes zero-distance to itself

    logging.debug("mean neighborhood size: {}".format(np.mean([x.shape[0] for x in neighbourhoods])))

    logging.debug("building sets..")
    return build_disjoint_sets(neighbourhoods)


def cluster_rec(
    df,
    cluster_col="cluster",
    vectors_col="embeddings",
    current_step=0,
    steps=3,
    threshold=5.0,
    threshold_step=1.2,
    batch_size=None,
    n_jobs=-1,
):
    # base
    if current_step >= steps:
        return df

    logging.debug(f"clustering step {current_step}..")

    current_vectors_col = f"{vectors_col}_{current_step}" if current_step > 0 else vectors_col
    current_cluster_col = f"{cluster_col}_{current_step}" if current_step > 0 else cluster_col
    next_vectors_col = f"{vectors_col}_{current_step + 1}"
    next_cluster_col = f"{cluster_col}_{current_step + 1}"

    centroids = df.groupby(current_cluster_col)[current_vectors_col].apply(np.mean).reset_index(name=next_vectors_col)

    centroids[next_cluster_col] = get_distance_sets(
        centroids[next_vectors_col].values.tolist(), threshold, batch_size, n_jobs
    )

    centroids = cluster_rec(
        centroids,
        cluster_col=cluster_col,
        vectors_col=vectors_col,
        current_step=current_step + 1,
        steps=steps,
        threshold=threshold * threshold_step,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    return pd.merge(df, centroids, on=current_cluster_col).drop(columns=[next_vectors_col])


def cluster_by_phash(df, args):
    logging.debug("clustering by phash..")

    logging.debug("loading hash..")
    total_hash_df = pd.read_csv(args.phash_csv).drop_duplicates(["item_image_name"])

    hash_df = total_hash_df[total_hash_df.item_image_name.isin(df.item_image_name)]

    logging.debug("group embeddings by hash..")
    df = pd.merge(df, hash_df, on="item_image_name", suffixes=("", "_tmp"))

    sample_df = df.groupby("phash").sample(random_state=42)

    # build clusters on phash
    sample_df["cluster"] = get_distance_sets(
        sample_df.embeddings.values.tolist(), args.threshold, args.batch_size, args.num_threads
    )

    # broadcast clusters back
    df = pd.merge(df, sample_df, on="phash", suffixes=("", "_tmp"))

    return df["cluster"]


def reduce_dim(embeddings: Sequence[Sequence[float]], n_components):
    n_samples, n_features = len(embeddings), len(embeddings[0])

    if not n_components or n_components >= min(n_samples, n_features):
        return embeddings
    logging.debug("applying pca..")

    pca = PCA(n_components=n_components)
    return list(pca.fit_transform(embeddings))


def check_clusters(clusters_csv_path="/tmp/clusters.csv", img_folder="/data/shared/aliimage_ru/"):
    df = pd.read_csv(clusters_csv_path)
    df["path"] = kutils.path.get_img_paths(img_folder, df.item_image_name)
    plot_clusters(df, depth=1, n_clusters=5, img_per_cluster=5)


def plot_clusters(df, depth=1, n_clusters=5, img_per_cluster=5):
    """depth=0 means split will be performed by initial clusters (which are probably just image ids)"""

    cluster_col = f"cluster_{depth}" if depth > 0 else "cluster"

    groups = [x for _, x in df.groupby(cluster_col)]
    random.shuffle(groups)

    print(f"Total data: {df.shape[0]}")
    print(f"Depth: {depth}")
    print(f"N clusters: {len(groups)}")

    i = 0
    for group in groups:
        if group.shape[0] < img_per_cluster:
            continue
        if i >= n_clusters:
            break
        i += 1

        sample = group.sample(img_per_cluster)

        fig, ax = plotters.im_grid(sample.path.values, nrows=1, ncols=len(sample.path), figsize=(14, 4))
        for axi in ax.flat:
            axi.axis("off")

        fig.tight_layout()
        plt.show()


def main():
    args = parse_args()

    logging_lvl = logging.DEBUG if args.verbose else logging.INFO
    setup_default_logging(logging_lvl, log_path="/var/log/cluster.log")

    logging.info(f"starting: {args.input_npz}")
    logging.debug(args)

    if osp.exists(args.output_csv):
        logging.info(f"{args.output_csv} already exists, terminating..")
        exit()

    # get embeddings
    tfz = np.load(args.input_npz)
    df = pd.DataFrame.from_dict(
        {
            "item_image_name": [osp.basename(x) for x in tfz["sample_ids"]],
            "embeddings": list(tfz["embeddings"]),
        }
    ).drop_duplicates(["item_image_name"])

    logging.info(f"{df.shape[0]} embeddings")

    if args.pca_dim:
        df.embeddings = reduce_dim(df.embeddings.values.tolist(), n_components=args.pca_dim)

    if args.phash_csv:
        df["cluster"] = cluster_by_phash(df, args)
    else:
        df["cluster"] = list(range(df.shape[0]))

    logging.debug("deep clustering..")

    # build clusters
    res = cluster_rec(df, steps=args.steps, threshold=args.threshold, threshold_step=args.threshold_step)

    logging.debug(res.head())
    logging.debug(res.shape)

    for col in res.columns:
        if col.startswith("cluster"):
            logging.debug(f"number of clusters, {col}: {res[col].nunique()}")

    res.to_csv(args.output_csv, index=False)

    logging.info(f"finished: {args.output_csv}")


if __name__ == "__main__":
    main()
