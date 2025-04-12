#!/usr/bin/env python
import numpy as np
import time
import timeit
import torch
from functools import partial


def optional_import(package_name, extras=None):
    """
    Attempt to import a package. If not found, raise ImportError with styled message.

    Parameters:
        package_name : str
            Name to pass to `__import__()`
        extras : str or list of str, optional
            Extra install flag(s) for pip, e.g. 'gpu', 'plot'
    """
    if isinstance(extras, str):
        extras = [extras]
    try:
        return __import__(package_name)
    except ImportError:
        extras_msg = f"[{','.join(extras)}]" if extras else ""
        raise ImportError(
            f"[writhe_tools] 👉 To enable this feature, install: pip install writhe-tools{extras_msg}"
        )



def to_numpy(x: "digit or iterable"):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (int, float, str, np.int64, np.int32, np.float32, np.float64)):
        return np.array([x])
    if isinstance(x, list):
        return np.asarray(x)
    if isinstance(x, (map, filter, tuple)):
        return np.asarray(list(x))
    if isinstance(x, torch.Tensor):
        return x.numpy()


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

    def __exit__(self, exc_type, exc_value, trace_back):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Time elapsed : {self.interval} s")
        if exc_type is not None:
            print(f"The following error occurred while timing the function {exc_value}")
            return False

        return False

    @classmethod
    def execution_time(cls, function: callable, n: int=1000, args: dict = None ):
        """
        Execute a function n times and compute the average and standard deviation of the
        execution time.

        Args:
            function : (callable) The function to be executed
            n : (int, optional) The nuber of times to execute the function
            args : (dict, optional) The arguments that should be passed to the funciton,  if applicable
        """

        function = partial(function, **args) if args is not None else function

        execution_times = timeit.repeat(function, repeat=n, number=1)

        return np.mean(execution_times), np.std(execution_times)


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
