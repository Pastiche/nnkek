__author__ = "akaiashi"

import argparse
import multiprocessing as mp
import os.path as osp
import sys
from typing import Sequence

import pandas as pd
from disjoint_set import DisjointSet
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from timm.utils import *
from tqdm import tqdm

from nnkek.utils.math import cdist_batch_parallel, cdist_batch


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
        help="path to csv with fields image_name, phash. If provided first round clustering will be performed by phash",
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

    return parser.parse_args()


def build_disjoint_sets(neighbourhoods: Sequence[Sequence[int]]):
    """neighborhoods[i] is a neighbourhood of the element i which may or may not include the i-th element itself"""

    rec_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(rec_limit, len(neighbourhoods)))

    ds = DisjointSet()

    for element, neighbourhood in enumerate(tqdm(neighbourhoods)):
        for neighbour in neighbourhood:
            ds.union(element, neighbour)

    sys.setrecursionlimit(rec_limit)

    return [ds.find(x) for x in range(len(neighbourhoods))]


def get_distance_sets(vectors: Sequence[Sequence[float]], threshold=5.0, batch_size=None, n_jobs=-1):
    logging.info("computing distances..")
    vectors = np.asarray(vectors)
    logging.info(f"vectors mean: {vectors.mean()}, std: {vectors.std()}")

    if n_jobs == -1:
        n_jobs = mp.cpu_count() - 1

    if n_jobs != 1:
        if not batch_size:
            batch_size = vectors.shape[0] // n_jobs
        distances = cdist_batch_parallel(vectors, batch_size=batch_size, n_jobs=n_jobs)
    else:
        distances = cdist_batch(vectors, batch_size=batch_size) if batch_size else cdist(vectors, vectors)

    logging.info("getting neighbours..")
    neighbourhoods = [np.argwhere(x < threshold).flatten() for x in distances]  # includes zero-distance to itself
    logging.info("mean neighborhood size: ".format(np.mean([len(x) for x in neighbourhoods])))

    logging.info("building sets..")
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

    logging.info(f"clustering step {current_step}..")

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
    logging.info("clustering by phash..")

    logging.info("loading hash..")
    total_hash_df = pd.read_csv(args.phash_csv).drop_duplicates(["image_name"])
    hash_df = total_hash_df[total_hash_df.image_name.isin(df.image_name)]

    logging.info("group embeddings by hash..")
    df = pd.merge(df, hash_df, on="image_name", suffixes=("", "_tmp"))
    sample_df = df.groupby("phash").sample(random_state=42)

    sample_df.embeddings = (
        reduce_dim(sample_df.embeddings.values.tolist(), n_components=args.pca) if args.pca else sample_df.empeddings
    )

    # build clusters on phash
    sample_df["cluster"] = get_distance_sets(
        sample_df.embeddings.values.tolist(), args.threshold, args.batch_size, args.num_threads
    )

    # broadcast clusters back
    df = pd.merge(df, sample_df, on="phash", suffixes=("", "_tmp"))

    return df["cluster"]


def reduce_dim(embeddings: Sequence[Sequence[float]], n_components):
    if not n_components:
        return embeddings
    logging.info("applying pca..")

    pca = PCA(n_components=n_components)
    return list(pca.fit_transform(embeddings))


def main():
    setup_default_logging()
    args = parse_args()

    # get embeddings
    tfz = np.load(args.input_npz_path)
    df = pd.DataFrame.from_dict(
        {
            "image_name": [osp.basename(x) for x in tfz[args.image_name_key]],
            "embeddings": list(tfz[args.embeddings_key]),
        }
    ).drop_duplicates(["image_name"])

    logging.info(f"{df.shape[0]} embeddings")

    if args.phash_csv:
        df["cluster"] = cluster_by_phash(df, args)
    else:
        df.embeddings = reduce_dim(df.embeddings.values.tolist(), n_components=args.pca) if args.pca else df.embeddings
        df["cluster"] = list(range(df.shape[0]))

    logging.info(df.head())
    logging.info("deep clustering..")

    # build clusters
    res = cluster_rec(df, steps=args.steps, threshold=args.threshold, threshold_step=args.threshold_step)

    logging.info(res.head())
    logging.info(res.shape)

    for col in res.columns:
        if col.startswith("cluster"):
            logging.info(f"number of clusters, {col}: ", res[col].nunique())

    res.to_csv(args.output_csv, index=False)

    logging.info("finished")


if __name__ == "__main__":
    main()
