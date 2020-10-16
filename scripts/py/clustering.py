__author__ = "n01z3"

import argparse
import os.path as osp
import sys
import time
from glob import glob
from typing import Sequence

import numpy as np
import pandas as pd
from disjoint_set import DisjointSet
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm

from nnkek.utils.math import cdist_batch_parallel
from nnkek.utils.process import batch_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="predict/ru",
        help="image features folder with npz files with fields: sample_ids, embeddings (sample_ids are image names)",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="tables/clusters/ru",
        help="path folder with image features and categories",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--threshold_multiplier",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--phash_csv",
        type=str,
        default="tables/phash_ru.csv",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--pca_threshold",
        type=int,
        default=30000,
        help="number of rows exceeds this value, pca will be applied to embeddings",
    )

    parser.add_argument("--hard_pca_dim", type=int, default=400)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
    )

    parser.add_argument("--category_id", type=int, default=-1)

    return parser.parse_args()


def build_disjoint_sets(neighbourhoods: Sequence[Sequence[int]]):
    """neighborhoods[i] is a neighbourhood of the element i which may or may not include the i-th element itself"""

    rec_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(rec_limit, len(neighbourhoods)))

    ds = DisjointSet()

    for element, neighbourhood in enumerate(tqdm(neighbourhoods, total=len(neighbourhoods))):
        for neighbour in neighbourhood:
            ds.union(element, neighbour)

    sys.setrecursionlimit(rec_limit)

    return [ds.find(x) for x in range(len(neighbourhoods))]


def get_distance_sets(vectors: Sequence[Sequence[float]], threshold=5.0):
    print("computing distances..")
    vectors = np.asarray(vectors)

    distances = (
        cdist(vectors, vectors) if len(vectors) < 100000 else cdist_batch_parallel(vectors)
    )  # TODO: magic number!
    print("getting neighbours..")
    neighbourhoods = [np.argwhere(x < threshold).flatten() for x in distances]  # includes zero-distance to itself
    print("building sets..")
    return build_disjoint_sets(neighbourhoods)


def cluster_rec(
    df,
    cluster_col="cluster",
    vectors_col="embeddings",
    current_step=0,
    steps=3,
    threshold=5.0,
    threshold_multiplier=1.2,
):
    # base
    if current_step >= steps:
        return df

    print(f"clustering step {current_step}..")

    current_vectors_col = f"{vectors_col}_{current_step}" if current_step > 0 else vectors_col
    current_cluster_col = f"{cluster_col}_{current_step}" if current_step > 0 else cluster_col
    next_vectors_col = f"{vectors_col}_{current_step + 1}"
    next_cluster_col = f"{cluster_col}_{current_step + 1}"

    centroids = df.groupby(current_cluster_col)[current_vectors_col].apply(np.mean).reset_index(name=next_vectors_col)

    centroids[next_cluster_col] = get_distance_sets(centroids[next_vectors_col].values.tolist(), threshold)

    centroids = cluster_rec(
        centroids,
        cluster_col=cluster_col,
        vectors_col=vectors_col,
        current_step=current_step + 1,
        steps=steps,
        threshold=threshold * threshold_multiplier,
    )

    return pd.merge(df, centroids, on=current_cluster_col).drop(columns=[next_vectors_col])


def cluster(predictions_path, args):

    # TODO сделать конфиг?
    # TODO отрефакторить)
    output_path, category_id = get_output_path(predictions_path, args)
    print(category_id)

    ts = time.time()

    # get hashes
    total_hash_df = pd.read_csv(args.phash_csv).drop_duplicates(["item_image_name"])

    # get embeddings
    tfz = np.load(predictions_path)
    embeddings_df = pd.DataFrame.from_dict(
        {"item_image_name": [osp.basename(x) for x in tfz["sample_ids"]], "embeddings": list(tfz["embeddings"])}
    ).drop_duplicates(["item_image_name"])

    # if args.verbose:
    print(embeddings_df.shape[0], " rows")

    # filter hashes
    hash_df = total_hash_df[total_hash_df.item_image_name.isin(embeddings_df.item_image_name)]
    embeddings_df = pd.merge(embeddings_df, hash_df, on="item_image_name", suffixes=("", "_tmp"))

    if args.verbose:
        print(embeddings_df.shape[0], " embeddings after merge on hash")

    print("loading data time: {}".format(time.time() - ts))
    ts = time.time()

    if args.verbose:
        print("clustering..")

    if args.verbose:
        print("group by hash..")

    # group by hash and assign initial clusters
    sample_df = embeddings_df.groupby("phash").sample(random_state=42)
    if args.verbose:
        print(f"{sample_df.shape[0]} rows after grouping by hash ")

    # TODO: исправить кудрявую логику здесь
    dim_reduction_coef = (sample_df.shape[0] // args.pca_threshold) + 1
    if dim_reduction_coef > 1 or args.hard_pca_dim:
        old_feats = sample_df.embeddings.values.tolist()
        pca_dim = len(old_feats[0]) // dim_reduction_coef
        if args.verbose:
            print(f"number of rows {sample_df.shape[0]} is too high, applying pca. dim={pca_dim}..")

        pca = PCA(n_components=args.hard_pca_dim if args.hard_pca_dim else pca_dim)
        sample_df.embeddings = list(pca.fit_transform(old_feats))

    if args.verbose:
        print("clustering by phash..")
    sample_df["cluster"] = get_distance_sets(sample_df.embeddings.values.tolist(), args.threshold)

    if args.verbose:
        print("deep clustering..")
    deep_clusters = cluster_rec(sample_df, steps=args.steps, threshold=args.threshold)

    res = pd.merge(embeddings_df, deep_clusters, on="phash", suffixes=("", "_tmp"))
    res.drop(columns=["embeddings", "embeddings_tmp", "item_image_name_tmp"], inplace=True)

    if args.verbose:
        print(res.head())
        print(res.shape)

        for col in res.columns:
            if col.startswith("cluster"):
                print(f"Number of clusters in {col}: ", res[col].nunique())

    print("clustering time: {}".format(time.time() - ts))
    ts = time.time()

    res.to_csv(output_path, index=False)

    print("dumping time: {}".format(time.time() - ts))


def get_output_path(input_path, args):
    category_id = osp.basename(input_path).split("_")[0]
    output_path = osp.join(args.output_folder, f"{category_id}_clusters.csv")
    return output_path, category_id


def main():
    args = parse_args()

    embeddings_paths = glob(osp.join(args.input_folder, "*"))

    if args.category_id != -1:
        embeddings_paths = [x for x in embeddings_paths if osp.basename(x).startswith(str(args.category_id))]

    print(f"Total files: {len(embeddings_paths)}")

    embeddings_paths = [x for x in embeddings_paths if not osp.exists(get_output_path(x, args)[0])]

    print(f"After skipping existing outputs: {len(embeddings_paths)}")
    print("Parallel clustering..")
    time.sleep(3)

    # TODO убрать вообще этот БАТЧ параллел здесь? И сделать, чтобы нельзя было cdist_parallel когда батчи
    # TODO сделать 2 версии - для одного файла параллельно и для набора файлов - каждый файл - в отдельном потоке,
    # хотя... вторая версия мб и не нужна! Нужен больше лишь тот скрипт, что я скидывал Артуру, но параллелизовнный нормально
    # а его уже запускать последовательно для каждой категории! да! Плюс привязка к категориям - слишком конретно

    if len(embeddings_paths) == 1:
        cluster(embeddings_paths[0], args=args)
    else:
        batch_parallel(embeddings_paths, cluster, batch_size=args.batch_size, n_jobs=args.num_threads, args=args)


if __name__ == "__main__":
    main()
