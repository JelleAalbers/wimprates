import functools
import inspect
import os
import pickle
import warnings

from boltons.funcutils import wraps
import numpy as np
from tqdm.autonotebook import tqdm


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


@export
def pairwise_log_transform(a, b):
    """
    Preprocesses two input arrays by reshaping, concatenating, and applying a logarithmic transformation.

    This function takes two input arrays (or single values), reshapes them into column vectors,
    concatenates them horizontally to form a 2D array, and then applies the natural logarithm
    to each element of the resulting array. It ensures compatibility with the downstream function
    that expects a 2D array with two columns, regardless of whether the input consists of single
    values or multiple pairs of values.

    Parameters
    ----------
    a : array-like or float
        The first input array or single float value.
    b : array-like or float
        The second input array or single float value.

    Returns
    -------
    numpy.ndarray
        A 2D array with shape (n, 2), where n is the number of elements in the input arrays.
        Each row contains the natural logarithm of the corresponding elements from the input arrays.

    Examples
    --------
    >>> pairwise_log_transform([4.5, 2.7], [400, 900])
    array([[ 1.5040774 ,  5.99146455],
           [ 0.99325177,  6.80239476]])

    >>> pairwise_log_transform(4.5, 400)
    array([[1.5040774 , 5.99146455]])

    Notes
    -----
    - If `a` and `b` are not arrays, they will be converted to arrays.
    - If `a` and `b` are single values, the output will be a 2D array with a single row.
    - The function applies `np.log` to each element in the concatenated array.
    """
    a = np.atleast_1d(a).reshape(-1, 1)
    b = np.atleast_1d(b).reshape(-1, 1)
    arr = np.concatenate((a, b), axis=1)
    return np.log(arr)

@export
def deprecated(reason):
    """
    This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.
    """
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__} ({reason}).",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return new_func
    return decorator