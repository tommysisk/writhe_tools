#!/usr/bin/env python

import numpy as np
import torch
import itertools
#from math import floor
from numpy_indexed import group_by as group_by_
import functools
from .misc import to_numpy
from typing import Optional, Union


def split_list(lst, n):
    # Determine the size of each chunk
    k, m = divmod(len(lst), n)  # k is the size of each chunk, m is the remainder

    # If the length of the list is less than n, adjust n
    if len(lst) < n:
        n = len(lst)

    # Split the list into n sublists, distributing the remainder elements equally
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def make_index_function(x: np.ndarray, y: np.ndarray):
    fxn = np.zeros(x.max() + 1)
    fxn[x] = y
    return np.vectorize(lambda i: fxn[i])


def zero_index(dtraj: np.ndarray):
    x = np.unique(dtraj)
    y = np.arange(x.size)
    return make_index_function(x, y)


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


def integrate_naturals(a: torch.Tensor,
                       b: torch.Tensor):
    """
    Sum up natural numbers from b to a with both a and b included in the sum
    (boundaries are inclusive on both sides)
    """
    return (a + b) * (b + 1 - a) // 2


def flat_index(i: torch.Tensor,
               j: torch.Tensor,
               n: int,
               m: int = -1,
               d0: int = 0,
               triu: bool = False,
               ):
    """
    retrieve index of data point in flattened matrix
    based on the indices the data point would have had in the unflattened matrix.

    If supdiagonals were removed from the original matrix before flattening, adjust
    the d0 input to account for it.
    d0=0 means that the main diagonal was included in the original flattening

    if the flattened matrix was upper triangular, use triu argument"""

    m = n if m == -1 else m
    if triu:
        # more obvious way of writting the subtraction part
        #    (sum of all naturals up to i + d0) - (sum of all the naturals up to, d0 - 1)
        #    thus, this is the sum of all naturals from d0 to i ...
        #    tri = (i + d0)
        #   (tri ** 2 + tri) // 2 - (d0 ** 2 - d0) // 2
        #
        return i * m + j - torch.floor_divide((2 * d0 + i) * (i + 1), 2)
    else:
        assert d0 == 1, "This function only works for d0=1 when triu=False, currently"
        return i * m + j - torch.floor_divide(i * (m - 1) + j, m).clamp(d0)


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
             reduction: callable = None,
             key_axis: int = 0):
    """
    Performs grouping of the values based on keys and can perform an operation
    on all items of a key's set of values. Is a generalized version of torch_scatter
    for numpy arrays and a few special cases ...

    If only a set of keys are passed (like the labels of a clustering),
    the function returns the sets of indices belonging to each key.

    If only a reduction value is given, sets the values to 1
    in order for this function to be used to count the number of
    times each key is seen.
    """
    keys = np.unique(keys, axis=key_axis, return_inverse=True)[-1] if keys.ndim > 1 else keys
    if reduction is not None:
        values = np.ones_like(keys) if values is None else values.squeeze()
        if values.ndim > 1:
            groups = [i[-1] for i in group_by_(keys=keys, values=values, reduction=reduction)]
            return np.stack(groups).squeeze() if len({i.shape for i in groups}) == 1\
                    else groups
        else:
            return np.asarray(group_by_(keys=keys, values=values, reduction=reduction))[:, -1]
    else:
        values = np.arange(len(keys)) if values is None else values
        return group_by_(keys).split_array_as_list(values)


def product(x: np.ndarray, y: np.ndarray):
    return np.asarray(list(itertools.product(x, y)))


def combinations(x):
    return np.asarray(list(itertools.combinations(x, 2)))


def shifted_pairs(x: np.ndarray, shift: int, ax: int = 1):
    return np.stack([x[:-shift], x[shift:]], ax)

def one_hot(labels, nlabels:int = None):
    labels = labels.flatten()
    np.put_along_axis((one_hot_ :=
                       np.zeros((labels.size, labels.max() + 1\
                                       if nlabels is None else nlabels), dtype=int)),
                      labels[:, None],
                      1,
                      axis=1,
                      )
    return one_hot_

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


def incidence_writhe_edges(num_nodes):
    """
    Generate a torch tensor of node pairs corresponding to the transformation |B| W_edge |B|^T,
    and expand W_flattened to match the node pairs for scatter operations.
    W_edge is the pairwise writhe matrix and B is the incidence matrix of the segments
    used in the computation of the writhe.

    Args:
        num_nodes (int): Number of nodes in the sequential graph.
        contextual but now removed:
            W_flattened (torch.Tensor): Flattened upper-triangular writhe values (excluding diagonal and first off-diagonal).

    Returns:
        node_pairs (torch.Tensor): A (2, num_pairs) tensor where each column represents a (node_i, node_j) pair.
        W_expanded (torch.Tensor): Expanded writhe values matching the node pairs for scatter operations.
    """
    num_edges = num_nodes - 1  # Sequential graph has (N-1) edges

    # Generate non-adjacent edge pairs
    edge_pairs = [(i, j) for i in range(num_edges) for j in range(i + 2, num_edges)]

    # Map edge pairs to node pairs using the incidence matrix
    node_pairs = []
    for edge_i, edge_j in edge_pairs:
        # Nodes associated with edges in a sequential graph
        node_i1, node_i2 = edge_i, edge_i + 1
        node_j1, node_j2 = edge_j, edge_j + 1

        # Each writhe value contributes to the four corresponding node pairs
        node_pairs.extend([(node_i1, node_j1), (node_i1, node_j2),
                           (node_i2, node_j1), (node_i2, node_j2)])

    # Convert to torch tensor
    node_pairs_tensor = torch.tensor(node_pairs, dtype=torch.long).T  # Shape: (2, num_pairs)

    # Expand W_flattened to match the node pairs for scatter operations
    #W_expanded = W_flattened.repeat_interleave(4)

    return node_pairs_tensor


def dx_indices_from_segments(segments: torch.LongTensor,
                             n: int,
                             d0: int,
                             triu: bool,
                             n_batches: int = 1):
    """
    Get the indices of the 4 displacement vectors
    needed to compute the writhe of each segment pair (first dim of segments).
    Output will be LongTensor with same dimension as segments.

    Args:
        segments:  (torch.LongTensor)
        n: (int)
        d0: (int)
        triu: (bool)

    """

    indices = torch.stack([flat_index(segments[:, i],
                                      segments[:, j],
                                      n=n,
                                      d0=d0,
                                      triu=triu)
                           for i, j in zip((0, 0, 0, 1, 1, 2),
                                           (1, 2, 3, 2, 3, 3))], 1).long()
    if n_batches == 1:
        return indices

    else:
        offset = n ** 2 if triu is False and d0 == 0 \
            else (flat_index((n - 1) - d0, (n - 1), n=n, d0=d0, triu=True) + 1
                  * (2 if not triu else 1))
        pass


def contiguous_bool(data: np.ndarray = None,
                    condition: "a python function that returns true when a condition on data is met" = None,
                    bools: np.ndarray = None) -> list:
    """returns a list of numpy arrays containing the indices
    (of the zeroth dim) of data where a condition is met contiguously"""

    assert ~all(i is None for i in (bools, condition)), "Must provide condition or bools"

    if condition is not None:
        assert data is not None, "If a callable condition is given, must provide data to operate on"

    bools = np.insert(bools.astype(int) if bools is not None else \
                          condition(data).astype(int), 0, 0)

    idx = np.insert(np.arange(len(bools) - 1), 0, 0)

    comp = np.stack([idx, bools], axis=1)

    return [i[:, 0][1:] if len(i) > 1 else i[:, 0] for i in
            filter(lambda x: any(x[:, 1] != 0), np.split(comp, np.where(comp[:, 1] == 0)[0]))]
