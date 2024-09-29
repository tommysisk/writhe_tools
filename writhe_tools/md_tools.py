#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mdtraj as md
import warnings
from .utils import (save_dict,
                    load_dict,
                    to_numpy,
                    triu_flat_indices,
                    combinations,
                    product)


class ResidueDistances:
    def __init__(self,
                 index_0: np.ndarray = None,
                 traj=None,
                 index_1: np.ndarray = None,
                 chain_id_0: str = None,
                 chain_id_1: str = None,
                 contact_cutoff=0.5,
                 args: "dict or path (str) to saved dict" = None):

        # restore the class from dictionary that it saves
        if args is not None:
            if isinstance(args, str):
                args = load_dict(args)
            assert isinstance(args, dict), "args must dict or path to saved dict"
            self.__dict__.update(args)

        # set up new instance, compute distances
        else:
            assert all(arg is not None for arg in (index_0, traj)), (
                "Minimum inputs are MDTraj object and index_0 (residue indices array)"
                " if an args dict saved by this class is not provided")
            self.contact_cutoff = contact_cutoff

            if index_1 is None:
                self.chain_id_0, self.chain_id_1 = [chain_id_0] * 2
                self.residues_0, self.residues_1 = [get_residues(traj, index_0)] * 2
                self.n = len(index_0)
                self.m = None
                self.prefix = "Intra"
                self.distances = residue_distances(traj, index_0)[0]

            else:
                self.chain_id_0, self.chain_id_1 = chain_id_0, chain_id_1
                self.residues_0, self.residues_1 = get_residues(traj, [index_0, index_1])
                self.n, self.m = map(len, (index_0, index_1))
                self.prefix = "Inter"
                self.distances = residue_distances(traj, index_0, index_1)[0]

        pass

    def save(self, file: str):
        args = {attr: getattr(self, attr) for attr in filter(lambda x: hasattr(self, x),
                                                             ["distances", "index_0", "index_1",
                                                              "chain_id_0", "chain_id_1", "residues_0",
                                                              "residues_1", "n", "m", "prefix",
                                                              "contact_cutoff"])}
        save_dict(file, args)

        return None

    @classmethod
    def load(cls, file):
        return cls(args=load_dict(file))

    def sub_diag(self, d):
        assert self.prefix == "Intra", "Must be intra molecular distances to subsample distances"
        return self.distances[:, triu_flat_indices(self.n, 1, d)]

    def matrix(self, contacts: bool = False, cut_off: float = None):

        cut_off = self.contact_cutoff if cut_off is None else cut_off

        if contacts:
            return to_contact_matrix(to_distance_matrix(self.distances, self.n, self.m),
                                     cut_off=cut_off)
        else:
            return to_distance_matrix(self.distances, self.n, self.m)

    def plot(self, index: "list, int, str" = None,
             contacts: bool = False,
             contact_cut_off: float = None,
             dscr: str = "",
             label_stride: int = 3,
             font_scale: int = 1,
             line_plot_args: dict = None,
             ax=None):

        if contacts:
            __dtype = "Contacts"
            unit = ""
        else:
            __dtype = "Distances"
            unit = " (nm)"

        if index is None:
            matrix = self.matrix(contacts, contact_cut_off).mean(0)
            __stype = "Average "

        elif isinstance(index, (int, float)):
            index = int(index)
            matrix = self.matrix(contacts, contact_cut_off)[index]
            __stype = ""
            dscr = f"Frame {index}"

        else:
            index = to_numpy(index).astype(int)
            matrix = self.matrix(contacts, contact_cut_off)[index].mean(0)
            __stype = "Average "

            if dscr == "":
                warnings.warn(("Taking the average over a subset of indices."
                               "The option, 'dscr', (type:str) should be set to provide \
                                a description of the indices. "
                               "Otherwise, the plotted data is ambiguous"))

        if dscr != "":
            dscr = f" : {dscr}"

        title = f"{__stype}{self.prefix}molecular {__dtype}{dscr}"
        cbar_label = f"{__stype}{__dtype}{unit}"

        matrix_plot_args = {"matrix": matrix,
                            "ylabel": self.chain_id_0,
                            "yticks": self.residues_0,
                            "xlabel": self.chain_id_1,
                            "xticks": self.residues_1,
                            "title": title,
                            "cbar_label": cbar_label,
                            "label_stride": label_stride,
                            "font_scale": font_scale,
                            "ax": ax}

        if line_plot_args is None:
            return plot_distance_matrix(**matrix_plot_args)

        else:
            line_plot_args["xticks"] = self.residues_1
            line_plot_args["x"] = np.arange(len(self.residues_1))
            line_plot_args["font_scale"] = 1
            line_plot_args["hide_title"] = True

            matrix_plot_args["font_scale"] = 1.2
            matrix_plot_args["hide_x"] = True
            matrix_plot_args["xlabel"] = None
            matrix_plot_args["xticks"] = None
            matrix_plot_args["aspect"] = "auto"

            build_grid_plot(matrix_plot_args, line_plot_args)

            return



def get_residues(traj,
                 indices: "iterable of arrays or array" = None,
                 atoms: bool = False,
                 cat: bool = False):
    indices = np.arange(len(traj.top.select("name CA"))) if indices is None else indices
    assert isinstance(indices, (np.ndarray, list)), "indices must be np.ndarray or list of np.ndarrays"

    func = (lambda index: traj.top.select(f"resid {int(index)}")) if atoms else (
            lambda index: str(traj.top.residue(int(index))))

    if isinstance(indices, np.ndarray):
        return func(indices) if len(indices) == 1 else \
            np.concatenate(list(map(func, indices))) if (cat and atoms) else \
            list(map(func, indices))
    else:
        return [np.concatenate(list(map(func, i))) if (len(i) != 1 and cat and atoms) else
                list(map(func, i)) if len(i) != 1 else func(i) for i in indices]


def to_distance_matrix(distances: np.ndarray,
                       n: int,
                       m: int = None,
                       d0: int = 1):
    assert (distances.ndim == 2), "Must input a flattened distance array (n,d)"

    # info about flattened distance matrix
    N, d = distances.shape

    # intra molecular distances
    if m is None:
        matrix = np.zeros([N] + [n] * 2)
        i, j = np.triu_indices(n, d0)
        matrix[:, i, j] = distances
        return matrix + matrix.transpose(0, 2, 1)

    else:
        assert d == n * m, \
            "Given dimensions (n,m) do not correspond to the dimension of the flattened distances"

        return distances.reshape(-1, n, m)


def residue_distances(traj,
                      index_0: np.ndarray,
                      index_1: np.ndarray = None):
    # intra distance case
    if index_1 is None:
        indices = combinations(index_0)
        return md.compute_contacts(traj, indices)[0], indices

    # inter distance case
    else:
        indices = product(index_0, index_1)
        return md.compute_contacts(traj, indices)[0], indices


def to_contact_matrix(distances: np.ndarray, cut_off: float = 0.5):
    return np.where(distances < cut_off, 1, 0)


def plot_distance_matrix(matrix: np.ndarray,
                         xlabel: str = None,
                         xlabel_rotation: float = None,
                         ylabel_rotation: float = 90,
                         ylabel: str = None,
                         xticks: "list or np.ndarray" = None,
                         yticks: "list or np.ndarray" = None,
                         xticks_rotation: float = 90,
                         yticks_rotation: float = None,
                         label_stride: int = None,
                         title: str = None,
                         cbar: bool = True,
                         cmap: str = "jet",
                         cbar_label: str = None,
                         cbar_label_rotation: float = -90,
                         vmin: float = None,
                         vmax: float = None,
                         alpha_lines: float = 0,
                         norm: bool = None,
                         aspect: str = None,
                         font_scale: float = 1,
                         ax=None,
                         hide_x: bool = False,
                         invert_yaxis: bool = True):

    assert matrix.ndim == 2, "Must be 2d matrix"
    n, m = matrix.shape
    args = locals()

    cmap = getattr(plt.cm, cmap)

    if ax is None:
        fig, ax = plt.subplots(1)

    s = ax.imshow(matrix, cmap=cmap, aspect=aspect,
                  norm=norm, vmax=vmax, vmin=vmin,
                  )
    ax.set_title(label=title, size=10 * font_scale)
    ax.tick_params(size=3 * font_scale, labelsize=7 * font_scale)

    ax.set_xlabel(xlabel=xlabel, size=10.2 * font_scale,
                  rotation=xlabel_rotation)

    ax.set_ylabel(ylabel=ylabel, size=10.2 * font_scale,
                  rotation=ylabel_rotation, labelpad=13 + font_scale)
    if cbar:
        cbar = plt.colorbar(s, ax=ax, label=cbar_label,
                            fraction=0.046, pad=0.02,
                            )

        cbar.set_label(cbar_label, rotation=cbar_label_rotation,
                       size=10 * font_scale, labelpad=12 + np.exp(font_scale))

        cbar.ax.tick_params(labelsize=8 * font_scale)

    for dim, key in zip([m, n], ["xticks", "yticks"]):

        val = args[key]
        if val is not None:
            assert len(val) == dim, f"{key} don't match matrix dimension"

            loc = np.arange(0, len(val))[::label_stride]
            val = val[::label_stride]

            _ = getattr(ax, f"set_{key}")(loc,
                                          val,
                                          rotation=args[f"{key}_rotation"])
    if hide_x:
        ax.tick_params(axis='x',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom=False,  # ticks along the bottom edge are off
                       top=False,  # ticks along the top edge are off
                       labelbottom=False,
                       size=0)

        ax.set_xlabel("", labelpad=0, size=0)

    if invert_yaxis:
        ax.invert_yaxis()

    return s


def lineplot1D(x, y,
               color: str = None,
               ls: str = None,
               lw: float = None,
               marker: str = None,
               mfc: str = None,
               mec: str = None,
               fill_color: str = None,
               fill_alpha: float = None,
               title: str = None,
               xlabel: "list or np.ndarray" = None,
               xlabel_rotation: float = None,
               ylabel_rotation: float = None,
               ylabel: "list or np.ndarray" = None,
               xticks: "list or np.ndarray" = None,
               yticks: "list or np.ndarray" = None,
               xticks_rotation: float = 90,
               yticks_rotation: float = None,
               ymin: float = None,
               ymax: float = None,
               xmin: float = None,
               xmax: float = None,
               label_stride: int = None,
               label: str = None,
               font_scale: float = 1,
               hide_title: bool = False,
               ax=None,
               ):
    args = locals()
    n, m = map(len, (y, x))

    if ax is None:
        fig, ax = plt.subplots(1)

    if hide_title:
        ax.set_title(label="", size=0)
    else:
        ax.set_title(label=title, size=13 * font_scale)

    ax.tick_params(size=3 * font_scale, labelsize=6 * font_scale)

    ax.set_xlabel(xlabel=xlabel, size=12 * font_scale,
                  rotation=xlabel_rotation, )

    ax.set_ylabel(ylabel=ylabel, size=12 * font_scale,
                  rotation=ylabel_rotation, labelpad=11 * font_scale)

    if len(list(filter(None, (ymin, ymax)))) != 0: ax.set_ylim(ymin, ymax)
    if len(list(filter(None, (xmin, xmax)))) != 0: ax.set_xlim(xmin, xmax)

    ax.margins(x=0, y=0)

    s = ax.plot(x, y, color=color, ls=ls,
                lw=lw, marker=marker,
                mfc=mfc, mec=mec, label=label)

    if label is not None:
        ax.legend()

    if fill_color is not None:
        ax.fill_between(x, y, color=fill_color, alpha=fill_alpha)

    for dim, key in zip([m, n], ["xticks", "yticks"]):
        val = args[key]
        if val is not None:
            assert len(val) == dim, f"{key} don't match matrix dimension"
            loc = np.arange(0, len(val))[::label_stride]
            val = val[::label_stride]
            _ = getattr(ax, f"set_{key}")(loc,
                                          val,
                                          rotation=args[f"{key}_rotation"],
                                          )

    return s


def build_grid_plot(matrix_args: dict,
                    line_args: dict,
                    size: int = 1.5):
    fig = plt.figure(figsize=(3 * size, 3.1 * size),
                     constrained_layout=True)
    grid = matplotlib.gridspec.GridSpec(nrows=6, ncols=3, figure=fig, hspace=.001, wspace=0.2,
                                        top=.92)

    ax0 = fig.add_subplot(grid[:-1, :])
    ax1 = fig.add_subplot(grid[-1, :], sharex=ax0)
    matrix_args["ax"] = ax0
    line_args["ax"] = ax1
    line_args["font_scale"] = 1
    line_args["hide_title"] = True
    matrix_args["font_scale"] = 1.2
    matrix_args["hide_x"] = True
    matrix_args["xlabel"] = None
    matrix_args["xticks"] = None
    matrix_args["aspect"] = "auto"
    plot_distance_matrix(**matrix_args)
    #fig.execute_constrained_layout()
    lineplot1D(**line_args)
    return None

