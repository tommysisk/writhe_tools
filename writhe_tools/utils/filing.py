#!/usr/bin/env python

import pickle
import os


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

