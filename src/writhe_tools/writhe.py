#!/usr/bin/env python
__author__ = "Thomas.R.Sisk@DartmouthCollege"

import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import matplotlib.text
import logging
import torch
import math
from joblib import Parallel, delayed
from typing import Optional, Union, List

from .utils.indexing import split_list, get_segments
from .utils.torch_utils import estimate_segment_batch_size, catch_cuda_oom
from .writhe_nn import writhe_segments
from .writhe_ray import writhe_segments_ray
from .writhe_numba import writhe_segments_numba
from .utils.filing import save_dict, load_dict
from .utils.misc import to_numpy, Timer
# from .stats import window_average, mean
from .plots import lineplot1D


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
def mean(x: np.ndarray, weights: np.ndarray = None, ax: int = 0):
    return x.mean(ax) if weights is None else (weights[:, None] * x).sum(ax) / weights.sum()

def window_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def calc_writhe_parallel(xyz: np.ndarray,
                         segments: np.ndarray,
                         use_cross: bool = True,
                         cpus_per_job=1,
                         cpu_method: str = "ray"):
    assert cpu_method in ["ray", "numba"], "method should be 'ray' or 'numba'."

    if cpu_method == "ray":
        return writhe_segments_ray(xyz=xyz,
                                   segments=segments,
                                   use_cross=use_cross,
                                   cpus_per_job=cpus_per_job)
    if cpu_method == "numba":
        return writhe_segments_numba(xyz=xyz,
                                     segments=segments)


def writhe_batches_cuda(xyz: torch.Tensor,
                        segments: torch.LongTensor,
                        use_cross: bool = True,
                        device: int = 0):
    xyz = xyz.to(device)
    result = torch.cat([writhe_segments(xyz, i, use_cross=use_cross).cpu() for i in segments], axis=-1).numpy() \
        if isinstance(segments, (list, tuple)) else writhe_segments(xyz=xyz,
                                                                    segments=segments,
                                                                    use_cross=use_cross).cpu().numpy()
    del xyz
    torch.cuda.empty_cache()
    return result


# noinspection PyArgumentList
@catch_cuda_oom
def calc_writhe_parallel_cuda(xyz: torch.Tensor,
                              segments: torch.LongTensor,
                              use_cross: bool = True,
                              batch_size: int = None,
                              multi_proc: bool = True) -> np.ndarray:
    batch_size = estimate_segment_batch_size(xyz) if batch_size is None else batch_size

    if batch_size > len(segments):
        return writhe_batches_cuda(xyz, segments, use_cross=use_cross, device=0)

    split = math.ceil(len(segments) / batch_size)
    chunks = torch.tensor_split(segments, split)

    if len(segments) < 5 * batch_size or torch.cuda.device_count() == 1 or not multi_proc:
        return writhe_batches_cuda(xyz, chunks, use_cross=use_cross, device=0)

    else:
        minibatches = split_list(chunks, torch.cuda.device_count())
        return np.concatenate(Parallel(n_jobs=-1)(
            delayed(writhe_batches_cuda)(xyz,
                                         segments=j,
                                         use_cross=use_cross,
                                         device=i) for i, j in enumerate(minibatches)),
            axis=-1)


def to_writhe_matrix(writhe_features, n_points, length):
    # assert not all(i is None for i in (n_points, length)), "Must provide either n_points (atoms) or segment length"
    # if n_points is not None:
    writhe_features = (np.expand_dims(writhe_features, 0) if writhe_features.ndim < 2
                       else writhe_features)
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

    Includes plotting methods and utilities for saving and loading computation results efficiently.
    """

    def __init__(
            self,
            xyz: Optional[np.ndarray] = None,
            args: Optional[dict] = None,
            **kwargs
    ) -> None:
        """
        Initialize the Writhe instance.

        Args:
            xyz (np.ndarray, optional): Coordinate matrix (Nframes, Natoms, 3).
            args (dict, optional): Additional arguments to initialize class attributes.
            kwargs: Arbitrary keyword arguments to initialize class attributes.
        """
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

    @staticmethod
    def compute_writhe_(xyz: np.ndarray,
                        segments: np.ndarray,
                        cpus_per_job: int,
                        cuda: bool,
                        cuda_batch_size: int,
                        multi_proc: bool,
                        use_cross: bool,
                        cpu_method: str = "ray"
                        ) -> np.ndarray:
        """
        Perform the writhe computation using either CPU or GPU parallelization.

        Args:
            xyz (np.ndarray): Coordinate matrix (Nframes, Natoms, 3).
            segments (np.ndarray): Indices defining the segments to compute writhe.
            cpus_per_job (int): Number of CPUs to allocate per batch (if not using GPU).
            cuda (bool): Whether to use CUDA-enabled GPU for computation.
            cuda_batch_size (int): Number of segments per batch for CUDA computation.
            multi_proc (bool): Whether to use multiprocessing.
            use_cross (bool): Whether to use cross products in the computation (most accurate, slower) or
                              dot products (less accurate for very small angles, faster).
                              When using double precision, the dot product method should be virtualy indistinguishable from
                              the cross product version.
            cpu_method (str) : cpu multiprocessing paradigm / framework - must be 'ray' or 'numba'
                                'ray' if faster and is best for large jobs that use multiple resources
                                'numba' is about an order of magnitude slower than ray.



        Returns:
            np.ndarray: Computed writhe features.
        """

        if cuda and torch.cuda.is_available():
            return calc_writhe_parallel_cuda(segments=torch.from_numpy(segments).long(),
                                             xyz=torch.from_numpy(xyz),
                                             batch_size=cuda_batch_size,
                                             multi_proc=multi_proc,
                                             use_cross=use_cross)
        else:
            if cuda:
                print("You tried to use CUDA but it's not available according to torch, defaulting to CPUs.")
            if multi_proc:
                return calc_writhe_parallel(segments=segments,
                                            xyz=xyz,
                                            cpus_per_job=cpus_per_job,
                                            cpu_method=cpu_method)
            else:
                warnings.warn("You are not using any multiprocessing or GPUs! "
                              "Multiprocessing on CPU is managed by ray or numba."
                              "ray handles big jobs best and is fastest!."
                              ".... using torch broadcast calculation (returns numpy")
                return writhe_segments(segments=torch.from_numpy(segments).long(),
                                       xyz=torch.from_numpy(xyz),
                                       use_cross=use_cross).numpy()

    def compute_writhe(self,
                       length: int = None,
                       segments: np.ndarray = None,
                       matrix: bool = False,
                       store_results: bool = True,
                       xyz: Optional[np.ndarray] = None,
                       n_points: Optional[int] = None,
                       speed_test: bool = False,
                       cpus_per_job: int = 1,
                       cuda: bool = False,
                       cuda_batch_size: Optional[int] = None,
                       multi_proc: bool = True,
                       use_cross: bool = True,
                       cpu_method: str = "ray"
                       ) -> Optional[dict]:
        """
        Compute writhe at the specified segment length.

        Args:
            length (int, optional): Segment length for computation.
            segments (np.ndarray, optional) : (n_segments, 4) int array specifying the segments to compute the writhe from.
            matrix (bool): Whether to generate a symmetric writhe matrix. Default: False.
            store_results (bool): Whether to store results in the class instance. Default: True.
            xyz (np.ndarray, optional): Coordinates to use for computation.
            n_points (int, optional): Number of points in the topology.
            speed_test (bool): Whether to perform a speed test without storing results.
            cpus_per_job (int): Number of CPUs to allocate per batch.
            cuda (bool): Whether to use CUDA for computation.
            cuda_batch_size (int, optional): Batch size for CUDA computation.
            multi_proc (bool): Whether to enable multiprocessing.

        Returns:
            dict: Results of the computation, including writhe features and segments.
        """
        assert (length is None) != (segments is None), ("Must provide either the length or the segments but not both."
                                                        "In general, only the length arg should be set.")

        assert not all(i is not None for i in (length, segments))
        if xyz is None:
            assert self.xyz is not None, \
                "Must instantiate instance with coordinate array (xyz) or provide it as argument"
            xyz = self.xyz

        if n_points is None:
            if self.n_points is not None:
                n_points = self.n_points
            else:
                n_points = xyz.shape[1]

        segments = get_segments(n=n_points, length=length) if length is not None else segments

        if speed_test:
            with Timer():
                _ = self.compute_writhe_(xyz=xyz, segments=segments,
                                         cpus_per_job=cpus_per_job,
                                         cuda=cuda, cuda_batch_size=cuda_batch_size,
                                         multi_proc=multi_proc, use_cross=use_cross,
                                         cpu_method=cpu_method)
            return None

        results = dict(length=length,
                       n_points=n_points,
                       n=len(xyz),
                       )

        results["writhe_features"] = self.compute_writhe_(xyz=xyz, segments=segments,
                                                          cpus_per_job=cpus_per_job,
                                                          cuda=cuda, cuda_batch_size=cuda_batch_size,
                                                          multi_proc=multi_proc, use_cross=use_cross,
                                                          cpu_method=cpu_method)

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

    def save(self,
             path: Optional[str] = None,
             dscr: Optional[str] = None,
             ) -> None:

        """
        Save the current writhe data to a file.

        Args:
            path (str, optional): Directory to save the file. Default: current directory.
            dscr (str, optional): Description to include in the filename.
        """

        self.check_data()

        if path is not None:
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            path = os.getcwd()

        keys = ["writhe_features", "n_points", "n", "length", "segments"]

        file = (f"{path}/writhe_data_dict_length_{self.length}" if dscr is None
                else f"{path}/{dscr}_writhe_data_dict_length_{self.length}") + ".pkl"

        save_dict(file, {key: getattr(self, key) for key in keys})

        return

    @classmethod
    def load(cls, file: str):
        """
        Arg:
            file : a pickled python dictionary saved by this class
        Return:
            An restored instance of this class with data retrieved from file
        """
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
               n_points: Optional[int] = None,
               length: Optional[int] = None,
               writhe_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convenience function for reindexing and sorting non-redundant
        writhe calculation into a symmetric matrix (redundant) for visualization.

        Args:
            n_points (Optional[int], optional): Number of points in each topology to estimate segments. Defaults to None.
            length (Optional[int], optional): Length of each segment for writhe calculation. Defaults to None.
            writhe_features (Optional[np.ndarray], optional): The writhe feature array to use. Defaults to None.

        Returns:
            np.ndarray: A symmetric matrix representing the writhe features.
        """

        self.check_data()

        n_points = n_points if n_points is not None else self.n_points
        writhe_features = writhe_features if writhe_features is not None else self.writhe_features
        length = length if length is not None else self.length
        return to_writhe_matrix(writhe_features, n_points, length)

    def plot_writhe_matrix(self,
                           index: Optional[Union[int, List[int], str, np.ndarray]] = None,
                           absolute: bool = False,
                           xlabel: Optional[str] = None,
                           ylabel: Optional[str] = None,
                           xticks: Optional[np.ndarray] = None,
                           yticks: Optional[np.ndarray] = None,
                           xticks_rotation: Optional[Union[float, None]] = 0,
                           yticks_rotation: Optional[Union[float, None]] = 0,
                           label_stride: int = 5,
                           dscr: Optional[str] = None,
                           font_scale: float = 1,
                           cmap: Optional[str] = None,
                           ax: Optional[plt.Axes] = None,
                           #rotation: Optional[float] = 90,
                           weights: Optional[np.ndarray] = None) -> None:
        """
        Plots the writhe matrix for visualizing writhe values in topological frames.

        This method provides a way to display a matrix of writhe values, optionally averaged across frames
        or for a specific subset of frames. The matrix can be visualized with absolute values or as signed writhe.

        Args:
            index (Optional[Union[int, List[int], str, np.ndarray]], optional): Frame index or indices to plot. Can be a single integer, list of integers, 'str', or numpy.ndarray. Defaults to None.
            absolute (bool, optional): If True, takes the absolute value of the writhe. Defaults to False.
            xlabel (Optional[str], optional): Label for the x-axis. Defaults to None.
            ylabel (Optional[str], optional): Label for the y-axis. Defaults to None.
            xticks (Optional[np.ndarray], optional): Array or list of tick labels for the x-axis. Defaults to None.
            yticks (Optional[np.ndarray], optional): Array or list of tick labels for the y-axis. Defaults to None.
            xticks_rotation (Optional[float], optional): Rotation (degrees) of xtick labels. Defaults to 0.
            yticks_rotation (Optional[float], optional): Rotation (degrees) of ytick labels. Defaults to 0.
            label_stride (int, optional): Interval to reduce tick labels for visualization. Defaults to 5.
            dscr (Optional[str], optional): Description for the subset of frames averaged, if applicable. Defaults to None.
            font_scale (float, optional): Scale factor for font sizes. Defaults to 1.
            cmap (Optional[str], optional): Colormap to use in plot, if None defaults to seismic or Reds is abs = True
            ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on. If None, a new figure is created. Defaults to None.

        Returns:
            None: Displays the plot using Matplotlib.

        Raises:
            AssertionError: If `index` is provided incorrectly or if ticks don't match the number of points used in writhe calculation.
        """

        self.check_data()

        args = locals()

        mat = self.writhe_features

        if absolute:
            mat = np.abs(mat)
            cmap = "Reds" if cmap is None else cmap
            cbar_label = "Absolute Writhe"
            norm = None

        # can't define norm until we know if it's a mean or not. if abs, then norm isn't needed
        if index is None:
            mat = mean(mat, weights=weights)
            title = "Average Writhe Matrix"
        else:
            assert index is not None, "If not plotting average, must specify index to plot"
            index = to_numpy(index).astype(int)
            if len(index) == 1:
                mat = mat[index.item()]
                title = f"Writhe Matrix: Frame {index.item()}"
            else:
                mat = mean(mat[index], weights)
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
            cmap = "seismic" if cmap is None else cmap
            cbar_label = "Writhe"

        if ax is None:
            fig, ax = plt.subplots(1)

        s = ax.imshow(self.matrix(writhe_features=mat).squeeze(), cmap=cmap, norm=norm)

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


            labels = labels[:-self.length][np.linspace(0,
                                                       self.n_points - self.length - 1,
                                                       (self.n_points - self.length - 1) // label_stride).astype(int)]

            ticks = np.linspace(0,
                                self.n_points - self.length - 1,
                                len(labels))


            _ = getattr(ax, f"set_{key}")(ticks=ticks,
                                          labels=labels,
                                          rotation=args[f"{key}_rotation"])



        ax.invert_yaxis()

        pass

    def plot_writhe_total(self,
                          absolute: Optional[bool] = False,
                          window: Optional[int] = None,
                          color: Optional[str] = 'indianred',
                          start: int = 0,
                          stop: int = -1,
                          indices: np.ndarray = None,
                          num_xticks: int = 8,
                          unit: str = None,
                          font_scale: float = 1,
                          xticks: np.ndarray = None,
                          x_magnitude: int = 0,
                          xlabel: str = None,
                          **kwargs,
                          ) -> None:
        """
        Plots the total absolute writhe across time steps.

        Args:
            absolute Optional[bool], optional): Whether to take the absolute value of the writhe.
            window (Optional[int], optional): The size of the window for moving average smoothing. Defaults to None.
            ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on. If None, a new figure is created. Defaults to None.

        Returns:
            None: Displays the plot using Matplotlib.
        """

        self.check_data()
        features = self.writhe_features[to_numpy(indices).astype(int)] \
            if indices is not None else self.writhe_features

        writhe_total = np.sum((abs(features[start, stop]) if absolute else self.writhe_features), axis=1)
        xticks = np.arange(len(writhe_total)) * 10 ** x_magnitude if xticks is None else xticks
        legend = None

        if window is not None:
            writhe_total = window_average(x=writhe_total, N=window)
            xticks = np.linspace(0, xticks[-window], len(writhe_total)).astype(int)#xticks[:-1 - window + 1] + (window - 1) / 2
            legend = f"Window Average Size : {window}"

        label_stride = len(xticks) // (num_xticks + 1)
        if xlabel is None:
            xlabel=f"Time Step ({unit})" if unit is not None else "Time Step"
            xlabel = xlabel + "\u2219" + fr"$10^{{{x_magnitude}}}$" if x_magnitude != 0 and unit is None else xlabel
        args = dict(y=writhe_total,
                    title=f"Total {'Absolute' if absolute else ''} Writhe" + f"\n(Segment Length : {self.length})",
                    xlabel=xlabel,
                    ylabel="Total Writhe",
                    label=legend,
                    xticks=xticks,
                    label_stride=label_stride,
                    font_scale=font_scale,
                    ylabel_rotation=90,
                    color=color,
                    )

        lineplot1D(**{**args, **kwargs})

        return

    def plot_writhe_per_segment(self,
                                absolute: Optional[bool] = False,
                                color: Optional[str] = 'indianred',
                                index: Optional[Union[int, List[int], str, np.ndarray]] = None,
                                xticks: Optional[List[str]] = None,
                                label_stride: int = 5,
                                dscr: Optional[str] = None,
                                ax: Optional[plt.Axes] = None) -> None:
        """
        Plots the total absolute writhe per segment across time steps.

        This method can either plot the average total writhe across frames, or plot the writhe for a specific frame (or set of frames).

        Args:
            absolute (Optional[bool], optional): Whether or not to take the absolute value of the writhe.
            index (Optional[Union[int, List[int], str, np.ndarray]], optional): Frame index or indices to plot. Can be a single integer, list of integers, 'str', or numpy.ndarray. Defaults to None.
            xticks (Optional[List[str]], optional): List of tick labels for the x-axis. Defaults to None.
            label_stride (int, optional): Interval for displaying tick labels. Defaults to 5.
            dscr (Optional[str], optional): Description for the subset of frames averaged, if applicable. Defaults to None.
            ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on. If None, a new figure is created. Defaults to None.

        Returns:
            None: Displays the plot using Matplotlib.

        Raises:
            AssertionError: If `index` is not specified when `ave` is False.
        """
        self.check_data()
        writhe_total = (abs(self.matrix()) if absolute else self.matrix()).sum(1)

        if index is None:
            data = writhe_total.mean(0)
            title = f"Average Total {'Absolute' if absolute else ''} Writhe Per Segment"

        else:
            assert index is not None, "If not plotting average, must specify index to plot"
            index = to_numpy(int(index) if isinstance(index, (float, str, int)) else index).astype(int)

            if len(index) == 1:
                data = writhe_total[index]
                title = f"Total {'Absolute' if absolute else ''} Writhe Per Segment: Frame {index}"
            else:
                data = writhe_total[index].mean(0)
                if dscr is None:
                    warnings.warn(("Taking the average over a subset of indices."
                                   "The option, 'dscr', (type:str) should be set to provide"
                                   " a description of the indices. "
                                   "Otherwise, the plotted data is ambiguous")
                                  )
                    title = f"Ensemble Averaged Total {'Absolute' if absolute else ''} Writhe Per Segment"
                else:
                    title = f"Average Total {'Absolute' if absolute else ''} Writhe Per Segment : {dscr}"

        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(data, color=color)
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
