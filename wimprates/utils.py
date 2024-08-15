import functools
import hashlib
import inspect
import os
import pickle
from typing import Any, Callable
import warnings

from boltons.funcutils import wraps
import numpy as np
from tqdm.autonotebook import tqdm

import wimprates as wr


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


def _generate_hash(*args, **kwargs):
    # Create a string with the arguments and module version

    args_str = wr.__version__ + str(args)

    # Add keyword arguments to the string
    args_str += "".join(
        [f"{key}{kwargs[key]}" for key in sorted(kwargs) if key != "progress_bar"]
    )

    # Generate a SHA-256 hash
    return hashlib.sha256(args_str.encode()).hexdigest()


@export
def save_result(func: Callable) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(
        *args, cache_dir: str="wimprates_cache", save_cache: bool=True, load_cache: bool=True, **kwargs
    ):
        # Define the cache directory
        CACHE_DIR = cache_dir

        # Generate the hash based on function arguments and module version
        func_name = func.__name__
        cache_key = _generate_hash(*args, **kwargs)

        # Define the path to the cache file
        cache_file = os.path.join(CACHE_DIR, f"{func_name}_{cache_key}.pkl")

        # Check if the result is already cached
        if load_cache and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                print("Loading from cache: ", cache_file)
                return pickle.load(f)

        # Compute the result
        result = func(*args, **kwargs)

        if save_cache:
            # Ensure cache directory exists
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
            
            # Save the result to the cache
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
                print("Result saved to cache: ", cache_file)

        return result

    return wrapper
