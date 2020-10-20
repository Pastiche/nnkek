import numpy as np
from scipy.spatial.distance import cdist

from nnkek.utils.process import batch_parallel, iter_batch, iter_batch_parallel


def cdist_batch(XA: np.ndarray, XB: np.ndarray = None, batch_size=1000):
    XB = XB or XA
    return iter_batch(XA, cdist, batch_size, XB=XB)


def cdist_iter_batch_parallel(XA: np.ndarray, XB: np.ndarray = None, batch_size=1000, n_jobs=-1):
    XB = XB or XA
    return iter_batch_parallel(XA, cdist, n_jobs, batch_size, XB=XB)


def cdist_batch_parallel(XA: np.ndarray, XB: np.ndarray = None, batch_size=1000, n_jobs=-1):
    XB = XB or XA
    return batch_parallel(XA, cdist, n_jobs, batch_size, XB=XB)
