import inspect
import os
import pickle

from boltons.funcutils import wraps
import numpy as np
from tqdm import tqdm


def exporter():
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter()
__all__ += 'exporter DATA_DIR'.split()


DATA_DIR = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))


@export
def data_file(path):
    """Convert path in wimprates' data directory to absolute path"""
    return os.path.join(DATA_DIR, 'data', path)


@export
def load_pickle(path):
    with open(data_file(path), mode='rb') as f:
        return pickle.load(f)


@export
def vectorize_first(f):
    """For-loop-vectorize the first argument of the function,
    preserving signature, docstring, etc, with optional progressbar
    """
    @wraps(f)
    def newf(xs, *args, **kwargs):
        if 'progress_bar' in kwargs:
            itr = tqdm
            del kwargs['progress_bar']
        else:
            def itr(x):
                return x

        if isinstance(xs, (list, np.ndarray)) and len(xs):
            return np.array(
                [f(x, *args, **kwargs)
                 for x in itr(xs)])
        return f(xs, *args, **kwargs)
    return newf
