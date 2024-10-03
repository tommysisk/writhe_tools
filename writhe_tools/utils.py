#!/usr/bin/env python

import numpy as np
import torch
import pickle
import itertools
import time
from numpy_indexed import group_by as group_by_
import functools
import gc


def split_list(lst, n):
    # Determine the size of each chunk
    k, m = divmod(len(lst), n)  # k is the size of each chunk, m is the remainder

    # If the length of the list is less than n, adjust n
    if len(lst) < n:
        n = len(lst)

    # Split the list into n sublists, distributing the remainder elements equally
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def reindex_list(unsorted_list: list, indices: "list or np.ndarray"):
    return list(map(unsorted_list.__getitem__, to_numpy(indices).astype(int)))


def sort_by_val_in(indices: np.ndarray,
                   value: np.ndarray,
                   max: bool = True):
    stride = -1 if max else 1
    return indices[np.argsort(value[indices])[::stride]]


def sort_indices_list(indices_list: list,
                      obs: "numpy array with values corresponding to indices",
                      max: bool = True):
    """Sort each array in a list of indices arrays based on their values in obs."""
    sort = functools.partial(sort_by_val_in, value=obs, max=max)
    return list(map(sort, indices_list))


def flat_index(i: "row idx",
               j: "column idx",
               n: "rows",
               m: "cols" = None,
               d0: "degree of diag before flattening" = 0,
               triu: bool = False,
               ):
    """retrieve index of data point in flattened matrix
    based on the indices the data point would have had in the unflattened matrix.

    In the case of flattened upper diagonal matrix :
    If supdiagonals were removed from the original matrix before flattening, adjust
    the d0 input to account for it.
    d0=0 means that the main diagonal was included in the original flattening"""

    if m is None:
        m = n

    if triu:
        tri = i + d0
    else:
        tri = 0
        assert d0 == 0, "if not a triu matrix, d0 should be 0"

    index = i * m + j - (tri ** 2 + tri) / 2

    return index

def triu_flat_indices(n: int, d0: int, d1: int = None):
    """convienience function for getting indices of values in
    a flattened trui matrix of supdiagonal degree, d0, corresponding to
    degree d1, thus, d1>d0 as this essentially ~downsamples~
    the data in the flattened matrix but cannot upsample it"""
    if d1 is not None:
        assert d1 >= d0, \
            "New degree must be equal or larger than old degree to return meaningful result"
    else:
        d1 = d0

    args = dict(zip(["i", "j"], np.triu_indices(n, d1)))
    return flat_index(**args, n=n, d0=d0, triu=True).astype(int)


def group_by(keys: np.ndarray,
             values: np.ndarray = None,
             reduction: callable = None):

    if reduction is not None:
        values = np.ones_like(keys) / len(keys) if values is None else values

        if values.squeeze().ndim > 1:

            return np.stack([i[-1] for i in group_by_(keys=keys, values=values, reduction=reduction)])

        else:
            return np.asarray(group_by_(keys=keys, values=values, reduction=reduction))[:, -1]

    values = np.arange(len(keys)) if values is None else values

    return group_by_(keys).split_array_as_list(values)


def product(x: np.ndarray, y: np.ndarray):
    return np.asarray(list(itertools.product(x, y)))


def combinations(x):
    return np.asarray(list(itertools.combinations(x, 2)))


def shifted_pairs(x: np.ndarray, shift: int, ax: int = 1):
    return np.stack([x[:-shift], x[shift:]], ax)


def get_segments(n: int = None,
                 length: int = 1,
                 index0: np.ndarray = None,
                 index1: np.ndarray = None,
                 tensor: bool = False):
    """
    Function to retrieve indices of segment pairs for various use cases.
    Returns an (n_segment_pairs, 4) array where each row (quadruplet) contains : (start1, end1, start2, end2)
    """

    if all(i is None for i in (index0, index1)):
        assert n is not None, \
            "Must provide indices (index0:array, (optionally) index1:array) or the number of points (n: int)"
        segments = combinations(shifted_pairs(np.arange(n), length)).reshape(-1, 4)
        segments = segments[~(segments[:, 1] == segments[:, 2])]
        return torch.from_numpy(segments).long() if tensor else segments

    else:
        assert index0 is not None, ("If providing only one set of indices, must set the index0 argument \n"
                                    "Cannot only supply the index1 argument (doesn't make sense in this context")
        if index1 is not None:
            segments = product(*[shifted_pairs(i, length) for i in (index0, index1)]).reshape(-1, 4)
            return torch.from_numpy(segments).long() if tensor else segments
        else:
            segments = combinations(shifted_pairs(index0, length)).reshape(-1, 4)
            segments = segments[~(segments[:, 1] == segments[:, 2])]
            return torch.from_numpy(segments).long() if tensor else segments


def to_numpy(x: "int, list or array"):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (int, float, np.int64, np.int32, np.float32, np.float64)):
        return np.array([x])
    if isinstance(x, list):
        return np.asarray(x)
    if isinstance(x, (map, filter, tuple)):
        return np.asarray(list(x))


def load_dict(file):
    with open(file, "rb") as handle:
        dic_loaded = pickle.load(handle)
    return dic_loaded


def save_dict(file, dict):
    with open(file, "wb") as handle:
        pickle.dump(dict, handle)
    return None


class Timer:
    """import time"""

    def __init__(self, check_interval: "the time (hrs) after the call method should return false" = 1):

        self.start_time = time.time()
        self.interval = check_interval * (60 ** 2)

    def __call__(self):
        if abs(time.time() - self.start_time) > self.interval:
            self.start_time = time.time()
            return True
        else:
            return False

    def time_remaining(self):
        sec = max(0, self.interval - abs(time.time() - self.start_time))
        hrs = sec // (60 ** 2)
        mins_remaining = (sec / 60 - hrs * (60))
        mins = mins_remaining // 1
        secs = (mins_remaining - mins) * 60
        hrs, mins, secs = [int(i) for i in [hrs, mins, secs]]
        print(f"{hrs}:{mins}:{secs}")
        return None

    # for context management
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Time elapsed : {self.interval} s")
        return self.interval

def cleanup():
    gc.collect()  # Clean up unreferenced memory
    with torch.no_grad():
        torch.cuda.empty_cache()


def window_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def profile_function(algorithm, *args, track_gpu=False, device=None):
    """
    Function to profile execution time and GPU memory usage for functions with PyTorch tensors.

    returns : executation_time (seconds), max_gpu_memory_used (GB)

    """
    # Reset GPU memory stats (if using GPU)
    if track_gpu and device and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Time execution
    start_time = time.time()
    result = algorithm(*args)  # Execute the function
    end_time = time.time()

    # Track GPU memory after execution (if using GPU)
    if track_gpu:
        torch.cuda.synchronize()  # Make sure all GPU operations are done
        max_gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2) / 1e3  # Peak GPU memory
    else:
        gpu_memory_used = 0
        max_gpu_memory_used = 0

    # Compute execution time
    execution_time = end_time - start_time

    return execution_time, max_gpu_memory_used

def gpu_stats():
    def list_tensors_on_gpu():
        all_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
        for tensor in all_tensors:
            print(f"Tensor ID: {id(tensor)}, Size: {tensor.size()}, Device: {tensor.device}")
    list_tensors_on_gpu()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")
    # Total memory reserved by PyTorch
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Reserved memory: {reserved_memory / (1024 ** 2):.2f} MB")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))




