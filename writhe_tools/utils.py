#!/usr/bin/env python

import numpy as np
import torch
import pickle
import itertools
import time
from numpy_indexed import group_by as group_by_
import functools
import gc
import re
from collections import OrderedDict
import warnings
import os
import scipy


def sort_strs(strs: list, max=False, indexed: bool = False):
    """ strs ::: a list or numpy array of strings.
        max ::: bool, to sort in terms of greatest index to smallest.
        indexed ::: bool, whether or not to filter out strings that don't contain digits.
                    if set to False and string list (strs) contains strings without a digit, function
                    will return unsorted string list (strs) as an alternative to throwing an error."""

    # we have to ensure that each str in strs contains a number otherwise we get an error
    assert len(strs) > 0, "List of strings is empty"
    check = np.vectorize(lambda s: any(map(str.isdigit, s)))

    if isinstance(strs, list):
        strs = np.array(strs)

    # the indexed option allows us to filter out strings that don't contain digits.
    ## This prevents an error
    if indexed:
        strs = strs[check(strs)]
        assert len(strs) > 0, "List of strings is empty after filtering strings without digits"

    # if indexed != True, then we don't filter the list of input strings and simply return it
    ##because an attempt to sort on indices (digits) that aren't present results in an error
    else:
        if not all(check(strs)):
            warnings.warn("Not all strings contain a number, returning unsorted input list to avoid throwing an error. "
                          "If you want to only consider strings that contain a digit, set indexed to True ")

            return strs

    get_index = np.vectorize(functools.partial(num_str, return_str=False, return_num=True))
    indices = get_index(strs).argsort()

    if max:
        return strs[np.flip(indices)]

    else:
        return strs[indices]


def lsdir(dir,
          keyword: "list or str" = None,
          exclude: "list or str" = None,
          match: callable = all,
          indexed: bool = False):
    """ full path version of os.listdir with files/directories in order

        dir ::: path to a directory (str), required
        keyword ::: filter out strings that DO NOT contain this/these substrings (list or str)=None
        exclude ::: filter out strings that DO contain this/these substrings (list or str)=None
        indexed ::: filter out strings that do not contain digits.
                    Is passed to sort_strs function (bool)=False"""

    if dir[-1] == "/":
        dir = dir[:-1]

    listed_dir = os.listdir(dir)

    listed_dir = filter_strs(listed_dir, keyword=keyword, exclude=exclude, match=match)

    return [f"{dir}/{i}" for i in sort_strs(listed_dir, indexed=indexed)]


def filter_strs(strs: list,
                keyword: "list or str" = None,
                exclude: "list or str" = None,
                match: callable = all):
    if keyword is not None:
        strs = keyword_strs(strs, keyword=keyword, exclude=False, match=match)

    if exclude is not None:
        strs = keyword_strs(strs, keyword=exclude, exclude=True, match=match)

    return strs


def keyword_strs(strs: list,
                 keyword: "list or str",
                 exclude: bool = False,
                 match: callable = all):
    if isinstance(keyword, str):
        filt = (lambda string: keyword not in string) if exclude else\
               lambda string: keyword in string
    else:
        filt = (lambda string: match(kw not in string for kw in keyword)) if exclude else\
               (lambda string: match(kw in string for kw in keyword))

    return list(filter(filt, strs))


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

    args = dict(zip(["i", "j"], np.triu_indices(n, d1)))
    return flat_index(**args, n=n, d0=d0, triu=True).astype(int)


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


def get_available_cuda_memory(device: int = 0):
    """return VRAM available on a device in GB"""

    assert torch.cuda.is_available(), "CUDA is not available"

    return (torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)) / 1024 ** 3


def estimate_segment_batch_size(n_samples: int):
    return int((700 *  29977 * get_available_cuda_memory()) / (7.79229736328125 * n_samples))


def num_str(s, return_str=True, return_num=True):
    s = ''.join(filter(str.isdigit, s))

    if return_str and return_num:
        return s, int(s)

    if return_str:
        return s

    if return_num:
        return int(s)


def multireplace(string, replacements, ignore_case=False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    if not replacements:
        # Edge case that'd produce a funny regex and cause a KeyError
        return string
    if ignore_case:
        def normalize_old(s):
            return s.lower()
        re_mode = re.IGNORECASE
    else:
        def normalize_old(s):
            return s
        re_mode = 0
    replacements = {normalize_old(key): val for key, val in replacements.items()}
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)
    pattern = re.compile("|".join(rep_escaped), re_mode)
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)


def plain_state_dict(d, badwords=["module.","model."]):
    replacements= dict(zip(badwords,[""]*len(badwords)))
    new = OrderedDict()
    for k, v in d.items():
        name = multireplace(string=k, replacements=replacements, ignore_case=True)
        new[name] = v
    return new


def load_state_dict(model, file):
    try:
        model.load_state_dict(plain_state_dict(torch.load(file)))

    #if we're trying to load from lightning state dict

    except:

        model.load_state_dict(plain_state_dict(torch.load(file)["state_dict"]))

    return model


def get_metrics(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()
    steps = {x.step for x in event_accumulator.Scalars("epoch")}
    epoch = list(range(len(steps)))
    train_loss, val_loss = [-np.array([x.value for x in event_accumulator.Scalars(_key) if x.step in steps]) for _key in
                            ["train_loss","val_loss"]]
    return np.array(epoch),train_loss, val_loss


def get_extrema(x, extend: float = 0):
    return [x.min() - extend, x.max() + extend]

def pmf1d(x: np.ndarray,
          bins: int,
          weights: np.ndarray = None,
          norm: bool = True,
          range: tuple = None):
    count, edge = np.histogram(x, bins=bins, weights=weights, range=range)
    p = count / count.sum() if norm else count
    idx = np.digitize(x, edge[1:-1])
    pi = p.flatten()[idx]
    return p, pi, idx, edge[:-1] + np.diff(edge) / 2


def pmfdd(arrays: "a list of arrays or N,d numpy array",
          bins: int,
          weights: np.ndarray = None,
          norm: bool = True,
          range: tuple = None,
          statistic: str = None):
    """each array in arrays should be the same length"""

    if statistic is None:
        statistic = "count" if weights is None else "sum"

    if isinstance(arrays, list):
        assert all(isinstance(x, np.ndarray) for x in arrays), \
            "Must input a list of arrays"
        arrays = [i.flatten() for i in arrays]
        assert len({len(i) for i in arrays}) == 1, "arrays are not all the same length"
        arrays = np.stack(arrays, axis=1)
    else:
        assert isinstance(arrays, np.ndarray), \
            "Must input a list of arrays or a single N,d array"
        arrays = arrays.squeeze()

    count, edges, idx = scipy.stats.binned_statistic_dd(arrays,
                                                        values=weights,
                                                        statistic=statistic,
                                                        bins=bins,
                                                        expand_binnumbers=True,
                                                        range=range)

    # if range is not None:
    #     idx = np.stack([np.digitize(value, edge[1:-1]) for edge, value in zip(edges, arrays.T)]) + 1

    idx = np.ravel_multi_index(idx - 1, tuple([bins for i in arrays.T]))

    p = count / count.sum() if norm else count
    pi = p.flatten()[idx]
    return p, pi, idx, (edge[:-1] + np.diff(edge) / 2 for edge in edges)


def pmf(x: "list of arrays or array",
        bins: int,
        weights: np.ndarray = None,
        norm: bool = True,
        range: tuple = None):
    """
    returns : p, pi, idx, bin_centers

    """
    if isinstance(x, np.ndarray):
        x = x.squeeze()
        if x.ndim > 1:
            return pmfdd(x, bins, weights, norm, range)
        else:
            return pmf1d(x, bins, weights, norm, range)
    if isinstance(x, list):
        if len(x) == 1:
            return pmf1d(x[0], bins, weights, norm, range)
        else:
            return pmfdd(x, bins, weights, norm, range)



def make_symbols():
    unicharacters = ["\u03B1",
                     "\u03B2",
                     "\u03B3",
                     "\u03B4",
                     "\u03B5",
                     "\u03B6",
                     "\u03B7",
                     "\u03B8",
                     "\u03B9",
                     "\u03BA",
                     "\u03BB",
                     "\u03BC",
                     "\u03BD",
                     "\u03BE",
                     "\u03BF",
                     "\u03C0",
                     "\u03C1",
                     "\u03C2",
                     "\u03C3",
                     "\u03C4",
                     "\u03C5",
                     "\u03C6",
                     "\u03C7",
                     "\u03C8",
                     "\u03C9",
                     "\u00C5"]
    keys = "alpha,beta,gamma,delta,epsilon,zeta,eta,theta,iota,kappa,lambda,mu,nu,xi,omicron,pi,rho,final_sigma,sigma,tau,upsilon,phi,chi,psi,omega,angstrom"
    return dict(zip(keys.split(","), unicharacters))


symbols = make_symbols().__getitem__
