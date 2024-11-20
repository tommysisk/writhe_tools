#!/usr/bin/env python
__author__ = "Thomas.R.Sisk@DartmouthCollege"

import os
import ray
import multiprocessing
import functools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import matplotlib.text
import logging
import torch
import math
from joblib import Parallel, delayed

from .utils.indexing import split_list, get_segments
from .utils.torch_utils import estimate_segment_batch_size, catch_cuda_oom
from .writhe_nn import writhe_segments
from .utils.filing import save_dict, load_dict
from .utils.misc import to_numpy, Timer
from .stats import window_average



class MplFilter(logging.Filter):
    def filter(self, record):
        if record.msg == "posx and posy should be finite values":
            return 0
        else:
            return 1


matplotlib.text._log.addFilter(MplFilter())


# Here, we consider the approach at is fastest with ray multiprocessing on CPUs!
# The more straight forward way of computing the writhe is implemented in writhe_nn,
# this computation is written specifically for ray and should not be used otherwise


def nnorm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def writhe_segment(segment=None, xyz=None):
    """
    Version of the writhe computation written specifically for CPU + ray parallelization.
    See writhe_tools.writhe_nn.wirthe_segments for cleaner implementation that is generally more efficient.
    We use this version because ray manages memory and parallelization for large computations.

    compute the writhe (signed crossing) of 2 segments for all frames (index 0) in xyz (xyz can contain just one frame)
    **provide both of the following**
    segment: numpy array of shape (4,) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
             array([seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point])
    xyz: numpy array of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons

    """
    xyz = xyz.unsqueeze(0) if xyz.ndim < 3 else xyz
    # broadcasting trick
    # negative sign, None placement and order are intentional, don't change without testing equivalent option
    dx = nnorm((-xyz[:, segment[:2], None] + xyz[:, segment[None, 2:]]).reshape(-1, 4, 3))

    # for the following, array broadcasting is (surprisingly) slower than list comprehensions
    # when using ray!! (without ray, broadcasting is faster).
    dots = np.stack([(dx[:, i] * dx[:, j]).sum(-1)
                     for i, j in zip([0, 0, 0, 1, 1, 2],
                                     [1, 2, 3, 2, 3, 3])], axis=-1)
    # indices
    u, v, h = [0, 4, 5, 1], \
              [4, 5, 1, 0], \
              [2, 3, 2, 3]

    # surface area from scalars

    theta = np.sum([np.arcsin(((dots[:, i] * dots[:, j] - dots[:, k])
                               / np.sqrt(abs(((1 - dots[:, i] ** 2) * (1 - dots[:, j] ** 2))).clip(1e-10))
                               ).clip(-1, 1)) for i, j, k in zip(u, h, v)], axis=0)

    signs = np.sign(np.sum(dx[:, 0] * np.cross(xyz[:, segment[3]] - xyz[:, segment[2]],
                                               xyz[:, segment[1]] - xyz[:, segment[0]], axis=-1),
                           axis=-1)).squeeze()

    wr = (theta * signs) / (2 * np.pi)

    return wr


# @ray.remote
def writhe_segments_along_axis(segments: np.ndarray, xyz: np.ndarray, axis: int = 1):
    """helper function for parallelization to compute writhe over chuncks of segments for all frames in xyz"""
    # noinspection PyTypeChecker
    return np.apply_along_axis(func1d=functools.partial(writhe_segment, xyz=xyz),
                               axis=axis, arr=segments)


# Ray parallelization is substantially faster than python multiprocessing
def calc_writhe_parallel(segments: np.ndarray,
                         xyz: np.ndarray,
                         cpus_per_job: int = 1) -> "Nframes by Nsegments np.ndarray":
    """parallelize writhe calculation by segment, avoids making multiple copies of coordinate (xyz) matrix using Ray shared memory"""
    # ray.init()

    xyz_ref = ray.put(xyz)  # reference to coordinates that all child processes can access without copying
    writhe_segments_along_axis_ref = ray.remote(writhe_segments_along_axis)
    chunks = np.array_split(segments, int(multiprocessing.cpu_count() / cpus_per_job))
    result = np.concatenate(ray.get([writhe_segments_along_axis_ref.remote(segments=chunk, xyz=xyz_ref)
                                     for chunk in chunks])).T.squeeze()
    ray.internal.free(xyz_ref)
    ray.shutdown()
    return result


def writhe_batches_cuda(xyz: torch.Tensor,
                        segments: torch.LongTensor,
                        device: int = 0):
    xyz = xyz.to(device)
    result = torch.cat([writhe_segments(xyz, i).cpu() for i in segments], axis=-1).numpy() \
        if isinstance(segments, (list, tuple)) else writhe_segments(xyz=xyz, segments=segments).cpu().numpy()
    del xyz
    torch.cuda.empty_cache()
    return result


# noinspection PyArgumentList


@catch_cuda_oom
def calc_writhe_parallel_cuda(xyz: torch.Tensor,
                              segments: torch.LongTensor,
                              batch_size: int = None) -> np.ndarray:
    batch_size = estimate_segment_batch_size(len(xyz)) if batch_size is None else batch_size

    if batch_size > len(segments):
        return writhe_batches_cuda(xyz, segments, 0)

    split = math.ceil(len(segments) / batch_size)
    chunks = torch.tensor_split(segments, split)

    if len(segments) < 5 * batch_size or torch.cuda.device_count() == 1:
        return writhe_batches_cuda(xyz, chunks, device=0)

    else:
        minibatches = split_list(chunks, torch.cuda.device_count())
        return np.concatenate(Parallel(n_jobs=-1)(
            delayed(writhe_batches_cuda)(xyz, j, i) for i, j in enumerate(minibatches)),
            axis=-1)


def to_writhe_matrix(writhe_features, n_points, length):
    n = len(writhe_features)
    writhe_matrix = np.zeros([n] + [n_points - length] * 2)

    indices = np.stack(np.triu_indices(n_points - length, 1), axis=0)
    indices = indices[:, abs(indices[0] - indices[1]) != length]

    writhe_matrix[:, indices[0], indices[1]] = writhe_features
    writhe_matrix += writhe_matrix.transpose(0, 2, 1)

    return writhe_matrix.squeeze()


def normalize_writhe(wr: np.ndarray, ax: int = None):
    return 2 * ((wr - wr.min(ax)) / (wr.max(ax) - wr.min(ax))) - 1


########################### Implementation of writhe computation ::: needs all of the functions and class(es) defined above ################################


class Writhe:
    """Implementation of parallelized writhe calculation that is combinatorially efficient.
    This class also contains plotting methods that can be used when store_results == True
    when using compute_writhe method (defaults to True).
    
    This class does not compute redundant values (zeros, nans and repeats by definition of writhe) BUT can organize
    and sort non-redundant calculation results into (redundant) symmetric matrices for visualization
    and adacency/connectivity matrices for graphs. As a result, it is highly recommended to use the
    save and load methods of this class so that large and redundant symmetric matrices are not saved
    to memory. The preferred work flow is to save results using the save method and restore this class
    from the resulting file using the load method.

    It is highly recommended to use this class to compute writhe when using this package to ensure
    correctness and maximum efficiency. Writhe should never be computed between all possible segments
    generated for a given length as this guarantees the calculation will be more than twice as expensive
    as compared to the combinatorially efficient approach implemented in this class.
    
    ARGS:
    xyz : coordinate matrix ::: type : np.ndarray ::: shape : (Nframes, Natoms, 3) :::
    assumes atoms are alpha carbons but don't need to be
    (adjust axis label arguments accordingly if they're provided as args to plotting methods)
    
    Most of the arguments of the methods for computing writhe are redundant when computing writhe
    for the same coordinates the class is instantiated with apart from the chosen length scale.
    The arguments are there for completeness and to allow a class instance to be used
    like a function so that the same class instance can be used to compute writhe at multiple length scales"""

    def __init__(self,
                 xyz: np.ndarray = None,
                 args: dict = None,
                 **kwargs):

        self.__dict__.update(kwargs)

        if args is not None:
            self.__dict__.update(args)

        else:
            self.xyz = xyz
            self.writhe_features = None
            self.length = None
            self.segments = None
            if xyz is not None:
                self.n, self.n_points = xyz.shape[:2]
            else:
                self.n_points, self.n = None, None

    def compute_writhe_(self,
                        xyz: np.ndarray,
                        segments: np.ndarray,
                        cpus_per_job: int,
                        cuda: bool,
                        cuda_batch_size: int,
                        multi_proc: bool):

        if cuda and torch.cuda.is_available():
            return calc_writhe_parallel_cuda(segments=torch.from_numpy(segments).long(),
                                             xyz=torch.from_numpy(xyz),
                                             batch_size=cuda_batch_size)
        else:
            if cuda: print("You tried to use CUDA but it's not available according to torch, defaulting to CPUs.")
            if multi_proc:
                return calc_writhe_parallel(segments=segments, xyz=xyz, cpus_per_job=cpus_per_job)
            else:
                warnings.warn("You are not using any multiprocessing! "
                              "Multiprocessing on CPU is managed by ray"
                              "and avoids issues with memory overflow.")
                return writhe_segments(segments=torch.from_numpy(segments).long(),
                                       xyz=torch.from_numpy(xyz)).numpy()

    def compute_writhe(self,
                       length: "Define segment size : CA[i] to CA[i+length], type : int",
                       matrix: "Make symmetric writhe matrix" = False,
                       store_results: "Bind calculation results to class for plotting" = True,
                       xyz: "Coordinates to use in writhe calculation (n, points/atoms, 3), type : np.ndarray" = None,
                       n_points: "Number of points in each topology used to estimate segments, type : int " = None,
                       speed_test: "only test the speed of the calculation and return nothing" = False,
                       cpus_per_job: int = 1,
                       cuda: bool = False,
                       cuda_batch_size: "number of segments to compute per batch if using cuda" = None,
                       multi_proc: bool = True
                       ):
        """
        All arguments apart from length are not required (can be left as default)
         when this class is instantiated with xyz coordinates.

        The matrix argument allow for symmetric writhe matrices to be constructed
        and returned by this method. However, these matrices can be automatically generated after
        the calculation at any time using the matrix method if calculation results are saved.
        Additionally, these matrices will be automatically generated but not saved into memory when using the
        plot writhe matrix method.

        WARNING: this class is designed to encourage workflows where redundant symmetric matrices
        only exist in memory transiently while plotting. There is almost never a good reason to
        clog memory with symmetric matrices. This is particular true when using writhe data
        as an INPUT for data science methods : passing redundant inputs/repeated data points to data science methods
        has no benefit and will needlessly make computations more expensive.

        WARNING: in addition to the above warning, explicitly putting symmetric writhe matrices into memory
        can cause kernels to break and memory to overflow needlessly.
         """

        if xyz is None:
            assert self.xyz is not None, \
                "Must instantiate instance with coordinate array (xyz) or provide it as argument"
            xyz = self.xyz

        if n_points is None:
            if self.n_points is not None:
                n_points = self.n_points
            else:
                n_points = xyz.shape[1]

        # compute (indices) of all segments of a given length
        segments = get_segments(n=n_points,
                                length=length)

        if speed_test:
            with Timer():
                _ = self.compute_writhe_(xyz, segments, cpus_per_job,
                                         cuda, cuda_batch_size, multi_proc)
            return None

        results = dict(length=length,
                       n_points=n_points,
                       n=len(xyz),
                       )

        results["writhe_features"] = self.compute_writhe_(xyz, segments, cpus_per_job,
                                                          cuda, cuda_batch_size, multi_proc)

        results["segments"] = segments

        # bind results to class to use plotting functions (takes more memory)
        if store_results:
            self.__dict__.update(results)

        # reorganize into symmetric matrix for visualizing
        if matrix:
            results["writhe_matrix"] = to_writhe_matrix(writhe_features=results["writhe_features"],
                                                        n_points=n_points,
                                                        length=length)

        return results

    def save(self, dir: str = None, dscr: str = None):

        self.check_data()

        if dir is not None:
            if not os.path.isdir(dir):
                os.makedirs(dir)
        else:
            dir = os.getcwd()

        keys = ["writhe_features", "n_points", "n", "length", "segments"]

        file = (f"{dir}/writhe_data_dict_length_{self.length}" if dscr is None
                    else f"{dir}/{dscr}_writhe_data_dict_length_{self.length}") + ".pkl"

        save_dict(file, {key: getattr(self, key) for key in keys})

        return

    @classmethod
    def load(cls, file):
        return cls(args=load_dict(file))

    @property
    def has_data(self):
        return self.writhe_features is not None

    def check_data(self):
        assert self.has_data, ("Must populate class with data before using this method"
                               " HINT : "
                               "Run compute_writhe method or instantiate from saved dictionary")
        return

    def matrix(self,
               n_points: "number of points in each topology to estimate segments, type : int" = None,
               length: int = None,
               writhe_features: np.ndarray = None):

        """convenience function for reindexing and sorting non-redundant
         writhe calculation into symmetric matrix (redundant) for visualization"""

        self.check_data()

        n_points = n_points if n_points is not None else self.n_points
        writhe_features = writhe_features if writhe_features is not None else self.writhe_features
        length = length if length is not None else self.length
        return to_writhe_matrix(writhe_features, n_points, length)

    def plot_writhe_matrix(self, ave=True, index: "int or list or str" = None,
                           absolute=False, xlabel: str = None, ylabel: str = None,
                           xticks: np.ndarray = None, yticks: np.ndarray = None,
                           label_stride: int = 5, dscr: str = None,
                           font_scale: float = 1, ax=None):

        self.check_data()

        args = locals()

        mat = self.matrix()

        if absolute:
            mat = abs(mat)
            cmap = "Reds"
            cbar_label = "Absolute Writhe"
            norm = None

        # can't define norm until we know if it's a mean or not. if abs, then norm isn't needed
        if (ave and (index is None)):
            mat = mat.mean(0)
            title = "Average Writhe Matrix"
        else:
            assert index is not None, "If not plotting average, must specify index to plot"
            index = to_numpy(index).astype(int)
            if len(index) == 1:
                mat = mat[index.item()]
                title = f"Writhe Matrix: Frame {index.item()}"
            else:
                mat = mat[index].mean(0)
                if dscr is None:
                    warnings.warn(("Taking the average over a subset of indices."
                                   "The option, 'dscr', (type:str) should be set to provide"
                                   " a description of the indices. "
                                   "Otherwise, the plotted data is ambiguous")
                                  )
                    title = "Ensemble Averaged Writhe Matrix"
                else:
                    title = f"Average Writhe Matrix: {dscr}"

        if not absolute:
            vmax = abs(mat).max()
            norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-1 * vmax, vmax=vmax)
            cmap = "seismic"
            cbar_label = "Writhe"

        if ax is None:
            fig, ax = plt.subplots(1)

        s = ax.imshow(mat.squeeze(), cmap=cmap, norm=norm)

        cbar = plt.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=10 * font_scale, labelpad=2 + np.exp(font_scale))
        cbar.ax.tick_params(labelsize=7 * font_scale)

        ax.set_title(label=f"{title} \n(Segment Length : {self.length})", size=9 * font_scale)

        ax.tick_params(size=3 * font_scale, labelsize=7 * font_scale)

        ax.set_xlabel(xlabel=xlabel, size=10.2 * font_scale,
                      labelpad=10 + np.sqrt(font_scale))
        ax.set_ylabel(ylabel=ylabel, size=10.2 * font_scale,
                      labelpad=10 + np.sqrt(font_scale))

        # ticks are handled with caution as the length chosen to compute writhe determines the proper tick labels
        for i, key in enumerate(["yticks", "xticks"]):
            if args[key] is not None:
                assert isinstance(args[key], (np.ndarray, list)), \
                    "ticks arguments must be list or np.ndarray"

                labels = to_numpy(args[key]).squeeze()

                assert self.n_points == len(labels), \
                    (f"{key} don't match the number of points used to compute writhe"
                     "The number of points (n_points) used to compute writhe should be equal to the number of tick labels."
                     "This method will correctly handle tick labels to account for the length used to compute writhe")

            else:
                labels = np.arange(0, self.n_points)

            rotation = 90 if key == "xticks" else None

            labels = labels[:-self.length][np.linspace(0,
                                                       self.n_points - self.length - 1,
                                                       (self.n_points - self.length - 1) // label_stride).astype(int)
            ]

            ticks = np.linspace(0,
                                self.n_points - self.length - 1,
                                len(labels))

            _ = getattr(ax, f"set_{key}")(ticks=ticks,
                                          labels=labels,
                                          rotation=rotation)

        ax.invert_yaxis()

        pass

    def plot_writhe_total(self, window=None, ax=None):

        self.check_data()

        writhe_total = abs(self.writhe_features).sum(1)

        if window is not None:
            data = window_average(x=writhe_total, N=window)
            legend = f"Window Averge Size : {window}"
        else:
            data = writhe_total
            legend = None

        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(data, color="red", label=legend)
        ax.set_title("Total Absolute Writhe" + f"\n(Segment Length : {self.length})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Total Writhe")

        if legend is not None:
            ax.legend()
        pass

    def plot_writhe_per_segment(self,
                                ave=True,
                                index=None,
                                xticks: list = None,
                                label_stride: int = 5,
                                dscr: str = None,
                                ax=None, ):

        self.check_data()
        writhe_total = abs(self.matrix()).sum(1)

        if (ave and (index is None)):
            data = writhe_total.mean(0)
            title = "Average Total Absolute Writhe Per Segment"

        else:
            assert index is not None, "If not plotting average, must specify index to plot"
            index = to_numpy(int(index) if isinstance(index, (float, str, int)) else index).astype(int)

            if len(index) == 1:
                data = writhe_total[index]
                title = f"Total Absolute Writhe Per Segment: Frame {index}"
            else:
                data = writhe_total[index].mean(0)
                if dscr is None:
                    warnings.warn(("Taking the average over a subset of indices."
                                   "The option, 'dscr', (type:str) should be set to provide"
                                   " a description of the indices. "
                                   "Otherwise, the plotted data is ambiguous")
                                  )
                    title = "Ensemble Averaged Total Absolute Writhe Per Segment"
                else:
                    title = f"Average Total Absolute Writhe Per Segment : {dscr}"

        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(data, color="red")
        ax.set_title(title + f"\n(Segment Length : {self.length})")
        ax.set_xlabel("Residue")
        ax.set_ylabel("Total Writhe")
        if xticks is not None:
            _ = ax.set_xticks(ticks=np.arange(0, self.n_points - self.length, label_stride),
                              labels=xticks[:self.n_points - self.length][::label_stride], rotation=45)
        pass
###########################################################################################################


# In[ ]:


#######usage#####
# trj = md.load(...)
# #get CA positions (xyz)
# xyz = trj.atom_slice(trj.topology.select("name CA")).xyz
# writhe = Writhe(xyz)
# writhe_data = writhe.compute_writhe(length = 4, matrix = True, adj_matrix = True, store_results = True, return_segments=True)
# writhe.plot_writhe_matrix()
# writhe.plot_writhe_matrix(index = 45)

####multiple chains (r1 and r2 crossings)#####
# r1r2_writhe = calc_writhe_parallel(get_segments(index0=np.arange(24), index1=np.arange(24, 48)), xyz=xyz)
