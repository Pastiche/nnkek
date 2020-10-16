import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from nnkek.utils.common import make_batches
from nnkek.utils.process import parallel_processor


def cdist_batch(XA: np.ndarray, XB: np.ndarray = None, batch_size=1000):
    if not XB:
        XB = XA

    batches = make_batches(XA, batch_size)
    res = []

    for batch in tqdm(batches):  # по сути дата лоадер
        batch_dists = cdist(batch, XB)
        res.append(batch_dists)

    return np.concatenate(res)


def cdist_batch_parallel(XA: np.ndarray, XB: np.ndarray = None, batch_size=1000, n_jobs=-1):
    if not XB:
        XB = XA

    batches = make_batches(XA, batch_size)
    res = parallel_processor(sequence=batches, worker=cdist, n_jobs=n_jobs, XB=XB)
    return res[0]
