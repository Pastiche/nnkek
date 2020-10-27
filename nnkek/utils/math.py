import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

from nnkek.utils.process import batch_parallel, iter_batch


def sk_dist(XA: np.ndarray, XB: np.ndarray = None, n_jobs=-1):
    if len(XA.shape) == 1:
        XA = XA.reshape(1, -1)

    if len(XB.shape) == 1:
        XB = XB.reshape(1, -1)

    return pairwise_distances(XA, XB, n_jobs=n_jobs)


def sp_dist(XA: np.ndarray, XB: np.ndarray = None):
    if len(XA.shape) == 1:
        XA = XA.reshape(1, -1)

    if len(XB.shape) == 1:
        XB = XB.reshape(1, -1)

    return cdist(XA, XB)


def dist_batch(XA: np.ndarray, XB: np.ndarray = None, batch_size=1000, n_jobs=-1):
    """Split initial matrix into batches and processes them one by one. Each batch is processed with n_jobs processes
    DEPRECATED. Proven to be VERY slow (as well as https://gist.github.com/rtavenar/a4fb580ae235cc61ce8cf07878810567)
    Here is only for reference.
    """
    if XB is None:
        XB = XA
    return iter_batch(XA, sk_dist, batch_size, XB=XB, n_jobs=n_jobs)


def dist_batch_parallel(XA: np.ndarray, XB: np.ndarray = None, batch_size=1000, n_jobs=-1):
    """Split initial matrix into batches and each batch with it's own process.
    DEPRECATED. Proven to be VERY slow (as well as https://gist.github.com/rtavenar/a4fb580ae235cc61ce8cf07878810567)
    Here is only for reference.
    """
    if XB is None:
        XB = XA
    return batch_parallel(XA, sp_dist, n_jobs, batch_size, XB=XB)
