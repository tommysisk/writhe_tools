#!/usr/bin/env python

import pickle
import os
import numpy as np
from .sorting import rm_path, lsdir


def load_dict(file):
    with open(file, "rb") as handle:
        dict_loaded = pickle.load(handle)
    return dict_loaded


def save_dict(file, dict):
    with open(file, "wb") as handle:
        pickle.dump(dict, handle)
    return None


def makedirs(path):
    if os.path.exists(path):
        return path
    else:
        os.makedirs(path)
        return path


def load_array_dir(dir: str,
                   keyword: "list or str" = None,
                   stack: bool = False,
                   load: callable = np.load):
    if load == np.load or "npy" in keyword:
        keyword = [".npy", keyword] if isinstance(keyword, str) else \
                  [".npy"] + keyword if isinstance(keyword, list) else \
                  ".npy"

        frmt = lambda x: rm_path(x).replace(".npy", "")

    else:
        frmt = lambda x: rm_path(x)


    if stack:
        return np.stack(list(map(load, lsdir(dir, keyword))))
    else:
        return {frmt(file): load(file) for file in lsdir(dir, keyword=keyword)}