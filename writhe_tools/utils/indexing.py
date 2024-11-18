#!/usr/bin/env python

import numpy as np
import torch
import itertools
from numpy_indexed import group_by as group_by_
import functools
from .misc import to_numpy


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
    """
    retrieve index of data point in flattened matrix
    based on the indices the data point would have had in the unflattened matrix.

    If supdiagonals were removed from the original matrix before flattening, adjust
    the d0 input to account for it.
    d0=0 means that the main diagonal was included in the original flattening

    if the flattened matrix was upper triangular, use triu argument"""

    if m is None:
        m = n
    if triu:
        tri = i + d0
        return i * m + j - (tri ** 2 + tri) / 2
    else:
        return i * m + j - (i + 1) * d0


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

    return flat_index(**dict(zip(["i", "j"], np.triu_indices(n, d1))),
                      n=n, d0=d0, triu=True).astype(int)


def indices_stat(indices_list: list,
                 obs: np.ndarray = None,
                 stat: callable = np.mean,
                 axis: int = None,
                 max: bool = True):
    """
    returns : values, indices
    """
    stride = -1 if max else 1
    if obs is None:
        values = np.fromiter(map(len, indices_list), float)
        indices = values.argsort()[::stride]
        return values, indices

    if axis is not None:
        stat = functools.partial(stat, axis=axis)

    values = np.array([stat(obs[i]) for i in indices_list])
    indices = values.argsort()[::stride]
    return values, indices


def group_by(keys: np.ndarray,
             values: np.ndarray = None,
             reduction: callable = None):
    if reduction is not None:
        values = np.ones_like(keys) / len(keys) if values is None else values.squeeze()
        return np.stack([i[-1] for i in group_by_(keys=keys, values=values, reduction=reduction)]) if values.ndim > 1 \
            else np.asarray(group_by_(keys=keys, values=values, reduction=reduction))[:, -1]
    else:
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
        assert index0 is not None, ("If providing only one set of indices,"
                                    "must set the index0 argument and let index1 argument default to None  \n"
                                    "Cannot only supply the index1 argument (doesn't make sense in this context")
        if index1 is not None:
            segments = product(*[shifted_pairs(i, length) for i in (index0, index1)]).reshape(-1, 4)
            return torch.from_numpy(segments).long() if tensor else segments
        else:
            segments = combinations(shifted_pairs(index0, length)).reshape(-1, 4)
            segments = segments[~(segments[:, 1] == segments[:, 2])]
            return torch.from_numpy(segments).long() if tensor else segments


def contiguous_bool(data: np.ndarray = None,
                    condition: "a python function that returns true when a condition on data is met" = None,
                    bools: np.ndarray = None) -> list:
    """returns a list of numpy arrays containing the indices
    (of the zeroth dim) of data where a condition is met contiguously"""

    assert ~all(i is None for i in (bools, condition)), "Must provide condition or bools"

    if condition is not None:
        assert data is not None, "If a callable condition is given, must provide data to operate on"

    bools = np.insert(bools.astype(int) if bools is not None else\
            condition(data).astype(int), 0, 0)

    idx = np.insert(np.arange(len(bools) - 1), 0, 0)

    comp = np.stack([idx, bools], axis=1)

    return [i[:, 0][1:] if len(i) > 1 else i[:, 0] for i in
            filter(lambda x: any(x[:, 1] != 0), np.split(comp, np.where(comp[:, 1] == 0)[0]))]