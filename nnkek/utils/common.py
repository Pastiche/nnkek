import resource
from functools import wraps
from time import time
from typing import Sequence, Any

import torch


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print("func:{} args: {}, {} took: {:04.2f} sec".format(f.__name__, args, kw, te - ts))
        print("{} took {:04.2f} sec".format(f.__name__, te - ts))
        return result

    return wrap


def make_batches(sequence: Sequence[Any], batch_size=None) -> Sequence[Sequence[Any]]:
    if not batch_size:
        batch_size = len(sequence) // 10
    if batch_size == 0:
        batch_size = 1

    return [sequence[i : i + batch_size] for i in range(0, len(sequence), batch_size)]


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))
