# noinspection PyInterpreter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain
from typing import Union, Optional, List
from matplotlib.patches import Rectangle
from matplotlib import gridspec



#from .stats import pmf
from .utils.indexing import to_numpy



def get_flat_bin_indices_2d(x,
                            y,
                            bins=(10, 10),
                            range=None,
                            out_of_bounds_val=-1):
    """
    Return the flattened bin index for each (x, y) pair, using binning identical to np.histogram2d.

    Parameters:
        x, y               : Arrays of coordinates
        bins               : Int or (nx, ny) tuple
        range              : ((xmin, xmax), (ymin, ymax)), optional
        out_of_bounds_val  : Integer assigned to out-of-bounds values (default: -1)

    Returns:
        flat_indices       : 1D array of flattened bin indices
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if isinstance(bins, int):
        bins = (bins, bins)

    # Use the same bin edge computation as np.histogram2d
    hist_dummy, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=range)

    # Use searchsorted, like np.histogram2d internally does
    x_bin = np.searchsorted(x_edges, x, side="right") - 1
    y_bin = np.searchsorted(y_edges, y, side="right") - 1

    # Clamp points on the right edge (included in last bin per np.histogram2d)
    x_bin[x == x_edges[-1]] = bins[0] - 1
    y_bin[y == y_edges[-1]] = bins[1] - 1

    # Create mask for valid points
    valid = (
        (x_bin >= 0) & (x_bin < bins[0]) &
        (y_bin >= 0) & (y_bin < bins[1])
    )

    # Compute flat bin index
    flat = np.full_like(x_bin, out_of_bounds_val, dtype=int)
    flat[valid] = np.ravel_multi_index((x_bin[valid], y_bin[valid]), dims=bins)

    return flat


def box_plot(values: np.ndarray,
             errors: np.ndarray = None,
             labels: np.ndarray = None,
             label: str = None,
             ylabel: str = None,
             xlabel: str = None,
             ymin: float = None,
             ymax: float = None,
             cmap: str = "viridis",
             color_height: bool = False,
             color_list: list = None,
             title: str = None,
             ax=None,
             figsize: tuple = (10, 6),
             font_scale: float = 5,
             label_stride: int = 1,
             error_capsize: float = 5,
             error_capthick: float = 1,
             error_linewidth: float = 0.5,
             error_color: str = 'gray',
             width: float = .95,
             linewidth: float = 1,
             edgecolor: str = "black",
             alpha: float = 1,
             trunc: float = 0,
             pre_trunc: float = 0,
             rotation: float = 0,
             vert: bool = True):
    values, errors = [i.squeeze() if i is not None and isinstance(i, np.ndarray) else i for i in [values, errors]]
    values = to_numpy(values)
    assert len(values.shape) == 1, "Need a set of values that can be squeezed to one dimension"

    nstates = len(values)
    labels = np.arange(1, nstates + 1).astype(str) if labels is None else labels
    assert len(labels) == nstates, "Length of values and labels must be the same"

    if color_height:
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
        cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        color_list = [cmap_fn(norm(val)) for val in values]
    else:
        color_list = plt.get_cmap(cmap)(np.linspace(pre_trunc, 1 - trunc, nstates))\
                     if color_list is None else color_list

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    x_pos = np.arange(nstates)

    if vert:
        ax.bar(x_pos,
               values,
               yerr=errors,
               ecolor=error_color,
               color=color_list,
               capsize=error_capsize,
               width=width,
               linewidth=linewidth,
               edgecolor=edgecolor,
               align="center",
               alpha=alpha,
               error_kw=dict(capthick=error_capthick, lw=error_linewidth),
               label=label)
        ax.set_xticks(ticks=x_pos[::label_stride], labels=labels[::label_stride])
        ax.set_xlabel(xlabel, size=6 * font_scale)
        ax.set_ylabel(ylabel, size=6 * font_scale)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlim([-.5, nstates - .5])
        ax.tick_params("x", labelrotation=rotation)
    else:
        ax.barh(x_pos,
                values,
                xerr=errors,
                ecolor=error_color,
                color=color_list,
                capsize=error_capsize,
                height=width,
                linewidth=linewidth,
                edgecolor=edgecolor,
                align="center",
                alpha=alpha,
                error_kw=dict(capthick=error_capthick, lw=error_linewidth),
                label=label)
        ax.set_yticks(ticks=x_pos[::label_stride], labels=labels[::label_stride])
        ax.set_ylabel(xlabel, size=6 * font_scale)
        ax.set_xlabel(ylabel, size=6 * font_scale)
        ax.set_xlim(left=ymin, right=ymax)
        ax.set_ylim([-.5, nstates - .5])
        ax.tick_params("y", labelrotation=rotation)

    ax.set_title(title, size=6 * font_scale)
    ax.tick_params("both", labelsize=6 * font_scale)

    return ax




def get_color_list(n_colors: int, cmap: str, trunc=0, pre_trunc=0):
    cmap = getattr(plt.cm, cmap)
    cl = [cmap(i) for i in range(cmap.N)]
    return [cl[i] for i in np.linspace(1 + pre_trunc, len(cl) - 1 - trunc, n_colors).astype(int)]


def truncate_colormap(cmap: str, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

def annotated_matrix_plot(matrix: np.ndarray,
                          error_matrix: np.ndarray = None,
                          color_matrix: np.ndarray = None,
                          title: str = "Transition Matrix",
                          xlabel=r"$State_{i}$",
                          ylabel=r"$State_{j}$",
                          unit: str = "",
                          cbar_label: str = r"P($State_{j}$|$State_{i}$)",
                          textcolor: str = "white",
                          cmap="viridis",
                          tick_labels: list = None,
                          xticks: list = None,
                          yticks: list = None,
                          xticks_rotation: list = 0,
                          val_text_size: int = 40,
                          err_text_size: int = 40,
                          vmin: float = None,
                          vmax: float = None,
                          decimals: int = 2,
                          font_scale: float = 1,
                          figsize: tuple = (10, 10),
                          round_int=False,
                          alpha: float = 1,
                          norm = None,
                          ax=None,
                          grid: bool = False,
                          grid_spacing: int = 1,
                          grid_alpha: float = 0.3,
                          grid_color: str = "black",
                          grid_linewidth: float = 0.8,
                          xticks_top: bool = False):   # <-- new argument
    """
    matrix: data to display (rectangular or square).
    error_matrix: optional error values of same shape.
    tick_labels: default labels (overridden by xticks/yticks if provided).
    xticks/yticks: optional labels for x/y axes.
    xticks_top: bool. If True, x-axis ticks will be shown on the top.
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    n, m = matrix.shape

    # Display matrix without stretching aspect ratio
    s = ax.imshow(matrix if color_matrix is None else color_matrix,
                  cmap=cmap, vmin=vmin,
                  vmax=vmax, alpha=alpha,
                  aspect='auto', norm=norm)

    # Handle ticks
    default_labels_x = [f"{i + 1}" for i in range(matrix.shape[1])]
    default_labels_y = [f"{i + 1}" for i in range(matrix.shape[0])]

    xtick_labels = xticks if xticks is not None else (tick_labels or default_labels_x)
    ytick_labels = yticks if yticks is not None else (tick_labels or default_labels_y)

    # Add value + error annotations
    for i in range(matrix.shape[1]):  # loop over x
        for j in range(matrix.shape[0]):  # loop over y
            if round_int:
                val = str(int(matrix[j, i]))
                if error_matrix is not None:
                    err = int(error_matrix[j, i])
            else:
                val = str(round(matrix[j, i], decimals))
                if error_matrix is not None:
                    err = str(round(error_matrix[j, i], decimals))

            ax.text(i, j,
                    f"{val + (' ' + unit if error_matrix is None else '')}",
                    va='center', ha='center',
                    color=textcolor, size=val_text_size * font_scale,
                    weight="bold")

            if error_matrix is not None:
                ax.text(i, j, "\n\n $\pm${}{}".format(err, unit),
                        va='center', ha='center', color=textcolor,
                        size=err_text_size * font_scale, weight="bold")

    # Axis labels and ticks
    ax.set_xticks(list(range(matrix.shape[1])), xtick_labels, size=35 * font_scale, rotation=xticks_rotation)
    ax.set_yticks(list(range(matrix.shape[0])), ytick_labels, size=35 * font_scale)
    ax.set_xlabel(xlabel, size=45 * font_scale)
    ax.set_ylabel(ylabel, size=45 * font_scale)

    if xticks_top:
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

    if grid:
        for i in range(0, n, grid_spacing):
            for j in range(0, m, grid_spacing):
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5),
                    width=grid_spacing,
                    height=grid_spacing,
                    edgecolor=grid_color,
                    facecolor='none',
                    lw=grid_linewidth,
                    alpha=grid_alpha
                ))

    # Automatic colorbar aligned to the matrix height
    divider = make_axes_locatable(ax)
    cb_width = axes_size.AxesY(ax, aspect=1./20)  # width = 1/20 of matrix height
    pad = axes_size.Fraction(0.2, cb_width)
    cax = divider.append_axes("right", size=cb_width, pad=pad)

    cb = ax.figure.colorbar(s, cax=cax)
    cb.set_label(cbar_label, size=40 * font_scale)
    cb.ax.tick_params(labelsize=30 * font_scale)

    ax.set_title(title, size=45 * font_scale)
    return ax



def plot_distance_matrix(matrix: np.ndarray,
                         xlabel: Optional[str] = None,
                         xlabel_rotation: Optional[float] = None,
                         ylabel_rotation: float = 90,
                         ylabel: Optional[str] = None,
                         xticks: Optional[Union[list, np.ndarray]] = None,
                         yticks: Optional[Union[list, np.ndarray]] = None,
                         xticks_rotation: float = 0,
                         yticks_rotation: Optional[float] = None,
                         label_stride: int = 1,
                         title: Optional[str] = None,
                         cbar: bool = True,
                         cmap: str = "jet",
                         cbar_label: Optional[str] = None,
                         cbar_label_rotation: float = -90,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None,
                         alpha: float = 1,
                         #alpha_lines: float = 0,
                         norm: Optional[plt.Normalize] = None,
                         aspect: Optional[str] = None,
                         font_scale: float = 1,
                         ax: Optional[plt.Axes] = None,
                         figsize: Optional[tuple] = None,
                         hide_x: bool = False,
                         hide_y: bool = False,
                         invert_yaxis: bool = True,
                         grid: bool = False,
                         grid_spacing: int = 1,
                         grid_alpha: float = 0.3,
                         grid_color: str = "black",
                         grid_linewidth: float = 0.8,
                         file: str = None,
                         dpi: float = 1000):

    assert matrix.ndim == 2, "Must be 2d matrix"
    n, m = matrix.shape
    args = locals()

    cmap = getattr(plt.cm, cmap)

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.figure

    s = ax.imshow(matrix, cmap=cmap, aspect=aspect,
                  norm=norm, vmax=vmax, vmin=vmin,
                  alpha=alpha)

    ax.set_title(label=title, size=10 * font_scale)
    ax.tick_params(size=3 * font_scale, labelsize=7 * font_scale)

    ax.set_xlabel(xlabel=xlabel, size=10.2 * font_scale,
                  rotation=xlabel_rotation)

    ax.set_ylabel(ylabel=ylabel, size=10.2 * font_scale,
                  rotation=ylabel_rotation, labelpad=13 + font_scale)

    if cbar:
        cbar = plt.colorbar(s, ax=ax, label=cbar_label,
                            fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label, rotation=cbar_label_rotation,
                       size=10 * font_scale, labelpad=12 + np.exp(font_scale))
        cbar.ax.tick_params(labelsize=8 * font_scale)

    for i, (key, n_points) in enumerate(zip(["yticks", "xticks"], (n, m))):
        if args[key] is not None:
            assert isinstance(args[key], (np.ndarray, list)), \
                "ticks arguments must be list or np.ndarray"
            labels = np.asarray(args[key]).squeeze()
            assert n_points == len(labels), (
                f"{key} don't match the number of points used to compute writhe"
                "The number of points (n_points) used to compute writhe should be equal to the number of tick labels."
            )
        else:
            labels = np.arange(0, n_points)
        if not label_stride == 1:
            labels = labels[np.linspace(0,
                                        n_points - 1,
                                        (n_points - 1) // label_stride).astype(int)]
        ticks = np.linspace(0,
                            n_points - 1,
                            len(labels))
        _ = getattr(ax, f"set_{key}")(ticks=ticks,
                                      labels=labels,
                                      rotation=args[f"{key}_rotation"])

    if hide_x:
        ax.tick_params(axis='x', which='both',
                       bottom=False, top=False,
                       labelbottom=False, size=0)
        ax.set_xlabel("", labelpad=0, size=0)

    if hide_y:
        ax.tick_params(axis='y', which='both',
                       left=False, right=False,     # use left/right instead of bottom/top
                       labelleft=False, size=0)     # use labelleft for y-axis
        ax.set_ylabel("", labelpad=0, size=0)

    if invert_yaxis:
        ax.invert_yaxis()

    # ✅ Pixel-aligned grid using Rectangle patches
    if grid:
        for i in range(0, n, grid_spacing):
            for j in range(0, m, grid_spacing):
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5),
                    width=grid_spacing,
                    height=grid_spacing,
                    edgecolor=grid_color,
                    facecolor='none',
                    lw=grid_linewidth,
                    alpha=grid_alpha
                ))
    if file is not None:
        fig.savefig(file, bbox_inches='tight', dpi=dpi)

    return s




def lineplot1D(y,
               x: np.ndarray = None,
               y1: np.ndarray = None,
               y2: np.ndarray = None,
               color: str = None,
               ls: str = None,
               lw: float = None,
               alpha: float = None,
               marker: str = None,
               mfc: str = None,
               mec: str = None,
               ms: float = 1,
               fill_color: str = None,
               fill_alpha: float = None,
               title: str = None,
               xlabel: "list or np.ndarray" = None,
               xlabel_rotation: float = None,
               ylabel_rotation: float = None,
               ylabel: "list or np.ndarray" = None,
               xticks: "list or np.ndarray" = None,
               yticks: "list or np.ndarray" = None,
               xticks_rotation: float = 0,
               yticks_rotation: float = None,
               ymin: float = None,
               ymax: float = None,
               xmin: float = None,
               xmax: float = None,
               label_stride: int = 1,
               label: str = None,
               font_scale: float = 1,
               hide_title: bool = False,
               figsize: tuple = None,
               yscale: str = 'linear',
               legend: bool = False,
               ax=None,
               top_xticks: "list or np.ndarray" = None,
               ):

    args = locals()

    x = np.arange(len(y)) if x is None else x

    n, m = map(len, (y, x))

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    if hide_title:
        ax.set_title(label="", size=0)
    else:
        ax.set_title(label=title, size=13 * font_scale)

    ax.tick_params(size=3 * font_scale, labelsize=12 * font_scale)

    ax.set_xlabel(xlabel=xlabel, size=12 * font_scale,
                  rotation=xlabel_rotation, )

    ax.set_ylabel(ylabel=ylabel, size=12 * font_scale,
                  rotation=ylabel_rotation, labelpad=11 * font_scale)

    if len(list(filter(None, (ymin, ymax)))) != 0: ax.set_ylim(ymin, ymax)
    if len(list(filter(None, (xmin, xmax)))) != 0: ax.set_xlim(xmin, xmax)

    ax.margins(x=0, y=0)

    s = ax.plot(x, y, color=color, ls=ls,
                lw=lw, marker=marker,
                mfc=mfc, mec=mec,
                label=label, ms=ms, alpha=alpha)

    if (label is not None) and legend:
        ax.legend()

    if fill_color is not None or all(i is not None for i in (y1, y2)):
        if all(i is not None for i in (y1, y2)):
            ax.fill_between(x, y1, y2, color=fill_color, alpha=fill_alpha)
        else:
            ax.fill_between(x, y, color=fill_color, alpha=fill_alpha)

    ax.set_yscale(yscale)

    # noinspection DuplicatedCode
    for i, (key, n_points) in enumerate(zip(["xticks"], [m])):
        if args[key] is not None:
            assert isinstance(args[key], (np.ndarray, list)), \
                "ticks arguments must be list or np.ndarray"
            labels = np.asarray(args[key]).squeeze()
            assert n_points == len(labels), (
                f"{key} don't match the number of points used to compute writhe"
                "The number of points (n_points) used to compute writhe should be equal to the number of tick labels."
            )
        else:
            labels = np.arange(0, n_points)
        if not label_stride == 1:
            labels = labels[np.linspace(0,
                                        n_points - 1,
                                        (n_points - 1) // label_stride).astype(int)]
        ticks = np.linspace(0,
                            n_points - 1,
                            len(labels))
        _ = getattr(ax, f"set_{key}")(ticks=ticks,
                                      labels=labels,
                                      rotation=args[f"{key}_rotation"])

    # Optional evenly spaced aesthetic labels on the TOP x-axis
    if top_xticks is not None:
        assert isinstance(top_xticks, (np.ndarray, list)), \
            "top_xticks must be list or np.ndarray"
        top_labels = np.asarray(top_xticks).squeeze()
        top_ticks = np.linspace(0, m - 1, len(top_labels))

        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(top_ticks, labels=top_labels, rotation=xticks_rotation)
        ax_top.tick_params(axis='x', size=3 * font_scale, labelsize=12 * font_scale)

    # for dim, key in zip([m, n], ["xticks", "yticks"]):
    #     val = args[key]
    #     if val is not None:
    #         assert len(val) == dim, f"{key} don't match matrix dimension"
    #
    #         labels = val[np.linspace(0,
    #                                  len(val) - 1,
    #                                  (len(val) - 1) // label_stride
    #                                  ).astype(int)]
    #
    #         ticks = np.linspace(0,
    #                             len(val) - 1,
    #                             len(labels))
    #
    #         _ = getattr(ax, f"set_{key}")(ticks,
    #                                       labels,
    #                                       rotation=args[f"{key}_rotation"],
    #                                       )

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
    # fig.execute_constrained_layout()
    lineplot1D(**line_args)
    return None


def fes2d(x: np.ndarray,
          y: np.ndarray = None,
          xlabel: str = None,
          ylabel: str = None,
          title: str = None,
          cbar: bool = True,
          cmap: str = "jet",
          vmax: float = None,
          vmin: float = None,
          cluster_centers=None,
          cluster_label_color: Optional[Union[str, List[str]]] = "black",
          cluster_label_size: Optional[float] = None,
          bins: int = 180,
          weights: np.ndarray = None,
          density: bool = False,
          extent: list = None,
          n_contours: int = 180,
          alpha_contours: float = 1,
          contour_lines: bool = False,
          alpha_lines: float = 0.6,
          scatter: bool = False,
          scatter_alpha: float = .2,
          scatter_cmap: str = "bone",
          scatter_size: float = 0.05,
          scatter_stride: int = 100,
          scatter_min: float = 0.2,
          scatter_max: float = 0.8,
          comp_type: str = None,
          mask: bool = True,
          font_scale: float = 1,
          cbar_shrink: float = 1,
          nxticks: int = 4,
          nyticks: int = 4,
          tick_decimals: int = 2,
          extend_border: float = 1e-5,
          hide_ax: bool = False,
          ax=None,
          aspect="auto",
          mask_thresh: float = 0,
          ):
    x, y = (np.squeeze(i) if i is not None else None for i in (x, y))

    if y is None:
        assert (x.ndim == 2) and (x.shape[-1] == 2), \
            ("Must provide 1d data vectors for x and y"
             "or provide x as a N,2 array with data vectors as columns")

        x, y = x.T

    if extent is None:
        extent = [[x.min() - extend_border, x.max() + extend_border],
                  [y.min() - extend_border, y.max() + extend_border]]

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=extent,
                                              weights=weights, density=density)

    # xticks, yticks = (i[:-1] + np.diff(i) / 2 for i in (x_edges, y_edges))

    mask_index = counts <= mask_thresh

    if mask:
        counts = np.ma.masked_array(counts, mask_index)

    else:
        counts[mask_index] = counts[~mask_index].min()

    F = -np.log(counts)
    F += -F.min()

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 3))

    # ax.margins(extend_border, tight=False)

    if extend_border == 0:
        ax.set_aspect(aspect=aspect)
        # extent = None

    s = ax.contourf(F.T,
                    levels=n_contours,
                    cmap=cmap,
                    extent=tuple(chain(*extent)),  # if extent is not None and extend_border != 0 else None,
                    zorder=-1,
                    alpha=alpha_contours,
                    vmax=vmax,
                    vmin=vmin
                    )

    if contour_lines:
        ax.contour(s, levels=s.levels, colors="black", cmap=None, linewidths=1, alpha=alpha_lines)

    fmtr = lambda x, _: f"{x:.2f}"

    xlabel = f"{comp_type} 1" if (comp_type is not None) and (xlabel is None) else xlabel
    ylabel = f"{comp_type} 2" if (comp_type is not None) and (ylabel is None) else ylabel

    ax.set_title(title, size=14 * font_scale)

    ax.set_xlabel(xlabel, fontsize=14 * font_scale)
    ax.set_ylabel(ylabel, fontsize=14 * font_scale)
    ax.tick_params(axis="x", labelsize=10 * font_scale)
    ax.tick_params(axis="y", labelsize=10 * font_scale)

    if extend_border != 0:
        ax.set_xlim(extent[0][0], extent[0][1])
        ax.set_ylim(extent[1][0], extent[1][1])

    ax.set_xticks(np.linspace(extent[0][0], extent[0][1], nxticks),
                  np.linspace(extent[0][0], extent[0][1], nxticks).round(tick_decimals))

    ax.set_yticks(np.linspace(extent[1][0], extent[1][1], nyticks),
                  np.linspace(extent[1][0], extent[1][1], nyticks).round(tick_decimals))

    if cbar:
        if contour_lines or n_contours < 8:
            cbar_ticks = s.levels[1:] - np.diff(s.levels) / 2

        else:
            cbar_ticks = np.linspace(s.levels[0], s.levels[-1], num=4, endpoint=True)

        cbar = plt.colorbar(s,
                            ax=ax,
                            format=fmtr,
                            shrink=cbar_shrink,
                            ticks=cbar_ticks)

        cbar.set_label("Free Energy / (kT)", size=12 * font_scale)
        cbar.ax.tick_params(labelsize=8 * font_scale)

    # ax.set_aspect(aspect=aspect, share=True)

    if scatter:
        c = F.flatten()[get_flat_bin_indices_2d(x, y, bins=bins)]
        # ax1 = ax.twinx().twiny()
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        ax.scatter(x[::scatter_stride], y[::scatter_stride],
                   cmap=truncate_colormap(scatter_cmap,
                                          minval=scatter_min,
                                          maxval=scatter_max,
                                          n=len(c[::scatter_stride])),
                   alpha=scatter_alpha,
                   c=c[::scatter_stride],
                   s=scatter_size)

        # ax1.autoscale_view()
        # if hide_ax:
        #     pass
        #     # ax1.set_axis_off()
        #     # ax1.axis("off")
        #     # ax.set_frame_on(False)

    if cluster_centers is not None:
        # ax2 = ax.twinx().twiny()
        # ax2.set_xticks([])
        # ax2.set_yticks([])

        cluster_label_color = len(cluster_centers) * [cluster_label_color] \
            if isinstance(cluster_label_color, str) \
            else cluster_label_color

        for j, (i, cc) in enumerate(zip(cluster_centers, cluster_label_color)):
            ax.annotate(f"{j + 1}", [i[k] for k in range(2)],
                        color=cc,
                        size=str(10 * font_scale) if cluster_label_size is None else cluster_label_size)
        # if hide_ax:
        #     pass
        #     # ax2.set_axis_off()
        #     # ax2.axis("off")
        #     # ax2.set_frame_on(False)

    if hide_ax:
        ax.set_axis_off()
        ax.axis("off")
        plt.gca().set_frame_on(False)
    # ax.autoscale_view()

    return s


def get_extrema(x, extend: float = 0):
    return [x.min() - extend, x.max() + extend]


def subplots_fes2d(x: np.ndarray,
                   rows: int,
                   cols: int,
                   dscrs: list,
                   indices_list: list = None,
                   #y: np.ndarray = None,
                   ylabel=None,
                   xlabel=None,
                   title: str = None,
                   title_pad: float = 1,
                   font_scale: float = .6,
                   cmap: str = "jet",
                   mask: bool = False,
                   mask_thresh: float = 0,
                   extent: list = None,
                   share_extent: bool = True,
                   sharex: bool = False,
                   sharey: bool = False,
                   n_contours: int = 200,
                   alpha_contours: float = 1,
                   contour_lines: bool = False,
                   alpha_lines: bool = 0.6,
                   bins: int = 100,
                   weights_list: list = None,
                   extend_border: float = 0,
                   density: bool = False,
                   scatter: bool = False,
                   scatter_alpha: float = .2,
                   scatter_cmap: str = "bone",
                   scatter_size: float = 0.05,
                   scatter_stride: int = 100,
                   scatter_min: float = 0.2,
                   scatter_max: float = 0.8,
                   figsize: tuple = (6, 5)):

    #x = np.stack([x, y], -1) if y is not None else x

    indices_list = list(range(len(x))) if indices_list is None else indices_list

    if extent is None:
        extent = [get_extrema(i, extend_border) for i in x.T] if isinstance(x, np.ndarray) \
                  else [get_extrema(i, extend_border) for i in np.concatenate(x).T] if share_extent \
                  else extent

    fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=sharex, figsize=figsize)

    if weights_list is None:
        for ax, indices, dscr in zip(axes.flat, indices_list, dscrs):
            s = fes2d(x[indices],
                      cbar=False,
                      cmap=cmap,
                      extent=extent,
                      mask=mask,
                      mask_thresh=mask_thresh,
                      bins=bins,
                      density=density,
                      n_contours=n_contours,
                      alpha_contours=alpha_contours,
                      contour_lines=contour_lines,
                      alpha_lines=alpha_lines,
                      title=dscr,
                      ax=ax,
                      font_scale=font_scale,
                      cbar_shrink=1,
                      extend_border=extend_border,
                      scatter=scatter,
                      scatter_alpha=scatter_alpha,
                      scatter_cmap=scatter_cmap,
                      scatter_size=scatter_size,
                      scatter_stride=scatter_stride,
                      scatter_min=scatter_min,
                      scatter_max=scatter_max,
                      )
    else:
        for ax, indices, dscr, weights in zip(axes.flat, indices_list, dscrs, weights_list):
            s = fes2d(x[indices],
                      cbar=False,
                      cmap=cmap,
                      extent=extent,
                      mask=mask,
                      mask_thresh=mask_thresh,
                      bins=bins,
                      density=density,
                      weights=weights,
                      n_contours=n_contours,
                      alpha_contours=alpha_contours,
                      contour_lines=contour_lines,
                      alpha_lines=alpha_lines,
                      title=dscr,
                      ax=ax,
                      font_scale=font_scale,
                      cbar_shrink=1,
                      extend_border=extend_border,
                      scatter=scatter,
                      scatter_alpha=scatter_alpha,
                      scatter_cmap=scatter_cmap,
                      scatter_size=scatter_size,
                      scatter_stride=scatter_stride,
                      scatter_min=scatter_min,
                      scatter_max=scatter_max,
                      )

    fig.subplots_adjust(right=1.05, top=.82, bottom=.18)
    fmtr = lambda x, _: f"{x:.1f}"
    c0, c1 = s.get_clim()
    cbar = fig.colorbar(s,
                        format=fmtr,
                        orientation='vertical',
                        ax=axes.ravel().tolist(),
                        aspect=20,
                        pad=.03,
                        panchor=(1, .5),
                        ticks=np.linspace(c0, c1, 4, endpoint=True)
                        )

    cbar.ax.tick_params(labelsize=12 * font_scale)
    cbar.set_label("Free Energy / (kT)", size=14 * font_scale)
    # x_offset = cols * cbar.ax.get_position().width / fig.get_size_inches()[0]
    # fig.supylabel(ylabel, x= 0 - x_offset, size=(100/6) * font_scale)
    # bbox = ax.get_xaxis().get_label().get_window_extent()
    # fig.supxlabel(xlabel,  x = .5 - x_offset,
    #               y=bbox.y0 / (fig.bbox.height) - font_scale / 12 -.08*np.exp(1.4 - figsize[-1]),
    #               size=(100/6) * font_scale)#y=-title_pad-0.01,
    # fig.suptitle(title, y=1+title_pad, x = .5 - x_offset, size=(100/6) * font_scale)
    fig.supylabel(ylabel, size=20 * font_scale, )
    fig.supxlabel(xlabel, size=20 * font_scale)
    fig.suptitle(title, size=20 * font_scale, y=title_pad)
    return


def sample_array(arr: np.ndarray, n, figs: int = 1):
    N = len(arr)
    return arr[np.round(np.linspace(n, N - n, n)).astype(int)].round(figs)


def proj2d(x: np.ndarray,
           c: np.ndarray,
           y: np.ndarray = None,
           xlabel: str = None,
           ylabel: str = None,
           title: str = None,
           cbar: bool = True,
           cbar_label: str = None,
           cbar_labels: str = None,
           cmap: str = "jet",
           cbar_norm=None,
           alpha: float = 1,
           cluster_centers: np.ndarray = None,
           center_font_color: str = "black",
           bins: int = 180,
           extent: list = None,
           comp_type: str = None,
           font_scale: float = 1,
           dot_size: float = 0.5,
           cbar_shrink: float = 1,
           nxticks: int = 4,
           nyticks: int = 4,
           tick_decimals: int = 2,
           vmin: float = None,
           vmax: float = None,
           ax=None,
           aspect="auto",
           state_map: bool = False,
           ):
    x, y = (np.squeeze(i) if i is not None else None for i in (x, y))

    if y is None:
        assert (x.ndim == 2) and (x.shape[-1] == 2), \
            ("Must provide 1d data vectors for x and y"
             "or provide x as a N,2 array with data vectors as columns")
        x, y = x.T

    if extent is None:
        extent = [[x.min(), x.max()], [y.min(), y.max()]]

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=extent)

    xticks, yticks = (i[:-1] + np.diff(i) / 2 for i in (x_edges, y_edges))

    fmtr = lambda x, _: f"{x:.2f}"

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 3))

    if cbar:
        if state_map:
            nstates = c.max() + 1
            color_list = getattr(plt.cm, cmap) if isinstance(cmap, str) else cmap
            boundaries = np.arange(nstates + 1).tolist()
            listed_colormap = matplotlib.colors.ListedColormap(
                [color_list(i) for i in range(color_list.N)])
            norm = matplotlib.colors.BoundaryNorm(boundaries, listed_colormap.N, clip=True)
            s = ax.scatter(x, y, c=c, s=dot_size, cmap=cmap, norm=norm, alpha=alpha)
            tick_locs = (np.arange(nstates) + 0.5)
            ticklabels = np.arange(1, nstates + 1).astype(str).tolist() \
                if cbar_labels is None else cbar_labels
            cbar = plt.colorbar(s, ax=ax, format=fmtr, shrink=cbar_shrink, )
            cbar.set_label(label="State" if cbar_label is None else cbar_label,
                           size=12 * font_scale)
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(ticklabels)

        else:
            s = ax.scatter(x, y, c=c, s=dot_size, cmap=cmap,
                           alpha=alpha, vmin=vmin, vmax=vmax)
            c0, c1 = s.get_clim()
            cbar = plt.colorbar(s, ax=ax, format=fmtr, shrink=cbar_shrink,
                                ticks=np.linspace(c0, c1, 4, endpoint=True),
                                norm=cbar_norm)
            cbar.set_label(cbar_label, size=12 * font_scale)

        cbar.ax.tick_params(labelsize=9 * font_scale)

    else:
        s = ax.scatter(x, y, c=c, s=.5, cmap=cmap,
                       alpha=alpha, vmin=vmin, vmax=vmax)

    ax.set_aspect(aspect)

    xlabel = f"{comp_type} 1" if (comp_type is not None) and (xlabel is None) else xlabel
    ylabel = f"{comp_type} 2" if (comp_type is not None) and (ylabel is None) else ylabel

    ax.set_title(title, size=14 * font_scale)

    ax.set_xlabel(xlabel, fontsize=14 * font_scale)
    ax.set_ylabel(ylabel, fontsize=14 * font_scale)

    ax.set_xticks(np.linspace(xticks[nxticks], xticks[bins - nxticks], nxticks),
                  sample_array(xticks, nxticks, figs=tick_decimals))
    ax.set_yticks(np.linspace(yticks[nyticks], yticks[bins - nyticks], nyticks),
                  sample_array(yticks, nyticks, figs=tick_decimals))

    # ax.set_xticks(np.linspace(nxticks, bins - nxticks, nxticks), sample_array(xticks, nxticks))
    # ax.set_yticks(np.linspace(nyticks, bins - nyticks, nyticks), sample_array(yticks, nyticks))

    ax.tick_params(axis="x", labelsize=10 * font_scale)
    ax.tick_params(axis="y", labelsize=10 * font_scale)

    if cluster_centers is not None:
        for j, i in enumerate(cluster_centers):
            ax.annotate(f"{j + 1}", [i[k] for k in range(2)],
                        color=center_font_color, size=str(10 * font_scale))

    return s


def subplots_proj2d(x: Optional[Union[np.ndarray, list]],
                    c: Optional[Union[np.ndarray, list]],
                    rows: int,
                    cols: int,
                    dscrs: list,
                    indices_list: list = None,
                    cmap: str = "jet",
                    dot_size: float = 0.5,
                    #y: np.ndarray = None,
                    ylabel=None,
                    xlabel=None,
                    title: str = None,
                    title_pad=0,
                    cbar_label: "str or list" = None,
                    font_scale: float = .6,
                    share_extent: bool = True,
                    sharey: bool = False,
                    sharex: bool = False,
                    vmin: float = None,
                    vmax: float = None,
                    bins: int = 100,
                    aspect: str = "auto",
                    figsize: tuple = None,
                    extent: tuple = None,
                    alpha=None
                    ):
    #x = np.stack([x, y], -1) if y is not None else x

    if indices_list is None:
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                indices_list = list(range(len(x)))
            else:
                # Same coordinates, different colors
                assert x.ndim == 2, "x must be 2 or 3 dimensional"
                indices_list = len(c) * [None]

        if isinstance(x, list):
            if len(x) > 1:
                indices_list = list(range(len(x)))


    if isinstance(c, np.ndarray):
        if c.ndim == 2:
            assert c.shape[0] == len(indices_list),\
                "If each plot has a different coloring, number must match number of datasets"
            color_indices = list(range(len(c)))
        else:
            # different coordinates, same colors
            assert c.ndim == 1, "c must be 1 or two dimensional"
            color_indices = len(indices_list) * [None]
    else:
        assert len(c) == len(indices_list) and isinstance(c, list), 'List of values for colors (c) must match length of indices_list'

        color_indices = list(range(len(c)))


    if extent is None:
        extent = [get_extrema(i) for i in x.T] if isinstance(x, np.ndarray) \
            else [get_extrema(i) for i in np.concatenate(x).T] if share_extent \
            else extent

    figsize = (2 * cols, 1.5 * rows) if figsize is None else figsize

    fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=sharex,
                             figsize=figsize, constrained_layout=False)

    # if isinstance(cbar_label, str):
    #     cbar_label = len(indices_list) * [cbar_label]
    # else:
    #     assert isinstance(cbar_label, list) and (len(cbar_label) == len(indices_list))
    c0, c1 = get_extrema(np.concatenate(c))
    c0, c1 = c0 if vmin is None else vmin, c1 if vmax is None else vmax

    for ax, indices, color_index, dscr in zip(axes.flat, indices_list, color_indices, dscrs):
        s = proj2d(x[indices],
                   c=c[color_index],
                   cmap=cmap,
                   dot_size=dot_size,
                   cbar=False,
                   cbar_label=cbar_label,
                   extent=extent,
                   bins=bins,
                   title=dscr,
                   ax=ax,
                   font_scale=font_scale,
                   vmin=c0,
                   vmax=c1,
                   cbar_shrink=1,
                   aspect=aspect,
                   alpha=alpha)
        # ax.set_aspect(aspect)

    fig.subplots_adjust(right=1.05, top=.82, bottom=.18)
    fmtr = lambda x, _: f"{x:.1f}"

    cbar = fig.colorbar(s,
                        format=fmtr,
                        orientation='vertical',
                        ax=axes.ravel().tolist(),
                        aspect=20,
                        pad=.03,
                        panchor=(1, .5),
                        ticks=np.linspace(c0, c1, 4, endpoint=True)
                        )

    cbar.ax.tick_params(labelsize=12 * font_scale)
    cbar.set_label(cbar_label, size=14 * font_scale)
    # x_offset = cols * cbar.ax.get_position().width / fig.get_size_inches()[0]
    # fig.supylabel(ylabel, x= 0 - x_offset, size=(100/6) * font_scale)
    # bbox = ax.get_xaxis().get_label().get_window_extent()
    # fig.supxlabel(xlabel,  x = .5 - x_offset, y=bbox.y0 / (fig.bbox.height) - font_scale / 12 -.08*np.exp(1.4 - figsize[-1]), size=(100/6) * font_scale)#y=-title_pad-0.01,
    # fig.suptitle(title, y=1+title_pad, x = .5 - x_offset, size=(100/6) * font_scale)
    fig.supylabel(ylabel, size=20 * font_scale)
    fig.supxlabel(xlabel, size=20 * font_scale)
    fig.suptitle(title, size=20 * font_scale, y = 1 + title_pad)
    return



def build_matrix_boxplot_grid(matrix: np.ndarray,
                              bottom_values: np.ndarray,
                              left_values: np.ndarray,
                              title: str = None,
                              title_size: int = 20,
                              left_title: str = None,
                              bottom_title: str = None,
                              marginal_max: int = None,
                              cmap: str = "jet",
                              figsize: tuple = (6, 6),
                              wspace: float = 0.05,
                              hspace: float = 0.05,
                              width_ratios: tuple = (1, 4, 0.15),
                              height_ratios: tuple = (1, 4, 1),
                              vmin: float = 0,
                              vmax: float = 1,
                              cbar_label: str = None,
                              ticks: list = None,
                              n_yticks: int = 3,
                              left_errors: np.ndarray = None,
                              bottom_errors: np.ndarray = None,
                              ecolor: str = 'gray',
                              outline: bool = True,
                              outline_color: str = "gray",
                              outline_lw: float = .8,
                              outline_alpha: float = 0.3,
                              path: str = None):
    """
    Heat‑map with marginal bar plots.

    * Matrix panel: rectangle outlines around every pixel (no internal grid).
    * Left / bottom bar panels: centred ticks only.
    * Optional: error bars and box border highlighting
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig,
                             width_ratios=width_ratios,
                             height_ratios=height_ratios,
                             wspace=wspace, hspace=hspace)

    # ---------------------------------------------------------------- Matrix
    ax_matrix = fig.add_subplot(spec[1, 1])
    im = ax_matrix.imshow(matrix, aspect="auto", cmap=cmap,
                          vmin=vmin, vmax=vmax, alpha=.85)
    ax_matrix.tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)
    ax_matrix.invert_yaxis()
    if title:
        ax_matrix.set_title(title, size=title_size)

    if outline:
        nrows, ncols = matrix.shape
        for i in range(nrows):
            for j in range(ncols):
                ax_matrix.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    edgecolor=outline_color,
                    facecolor="none",
                    lw=outline_lw,
                    alpha=outline_alpha
                ))

    nrows, ncols = matrix.shape

    # ---------------------------------------------------------- Left bar plot
    ax_left = fig.add_subplot(spec[1, 0], sharey=ax_matrix)
    bar_kwargs = dict(
        color="indianred",
        alpha=.85,
        edgecolor="black",
        linewidth=0.5
    )

    ax_left.barh(np.arange(nrows), left_values, **bar_kwargs)

    if left_errors is not None:
        ax_left.errorbar(left_values, np.arange(nrows), xerr=left_errors,
                         fmt='none', ecolor=ecolor, capsize=2, lw=0.8)

    left_max = max(vmax, bottom_values.max(), left_values.max()) if marginal_max is None else marginal_max
    ax_left.set_xlim(vmin, left_max)
    ax_left.invert_xaxis()
    ax_left.xaxis.tick_top()
    ax_left.yaxis.tick_right()
    ax_left.minorticks_off()

    xticks = np.linspace(vmin, left_max, 3)[:-1] if n_yticks == 2 else np.linspace(vmin, left_max, n_yticks)
    ax_left.set_xticks(xticks)
    ax_left.set_xticklabels(
        [f"{x:.0f}" if x.is_integer() else f"{x:.1f}" for x in xticks],
        fontsize=14
    )
    if left_title:
        ax_left.set_ylabel(left_title, rotation=90, labelpad=8, size=18)
        ax_left.yaxis.set_label_position("left")

    # -------------------------------------------------------- Bottom bar plot
    ax_bottom = fig.add_subplot(spec[2, 1], sharex=ax_matrix)
    bar_kwargs = dict(
        color="skyblue",
        alpha=.6,
        edgecolor="black",
        linewidth=0.5
    )

    ax_bottom.bar(np.arange(ncols), bottom_values, **bar_kwargs)

    if bottom_errors is not None:
        ax_bottom.errorbar(np.arange(ncols), bottom_values, yerr=bottom_errors,
                           fmt='none', ecolor=ecolor, capsize=2, lw=0.8)

    bottom_max = max(vmax, bottom_values.max(), left_values.max()) if marginal_max is None else marginal_max
    ax_bottom.set_ylim(vmin, bottom_max)
    ax_bottom.minorticks_off()
    ax_bottom.yaxis.tick_right()

    yticks = np.linspace(vmin, bottom_max, 3)[:-1] if n_yticks == 2 else np.linspace(vmin, bottom_max, n_yticks)
    ax_bottom.set_yticks(yticks)
    ax_bottom.set_yticklabels(
        [f"{y:.0f}" if y.is_integer() else f"{y:.1f}" for y in yticks],
        fontsize=14
    )
    if bottom_title:
        ax_bottom.set_xlabel(bottom_title, labelpad=8, size=18)
        ax_bottom.xaxis.set_label_position("bottom")

    # -------------------------------------------------------------- Colorbar
    cax = fig.add_subplot(spec[1, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    if cbar_label:
        cbar.set_label(cbar_label, rotation=-90, labelpad=16, size=16)
    cbar_ticks = np.linspace(im.norm.vmin, im.norm.vmax, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.0f}" if t.is_integer() else f"{t:.1f}" for t in cbar_ticks])

    # ---------------------------------------- Centre ticks on bar plots only
    pos_x = np.arange(ncols)
    pos_y = np.arange(nrows)
    ax_left.set_yticks(pos_y)
    ax_bottom.set_xticks(pos_x)

    if ticks is not None:
        ax_left.set_yticklabels(ticks,
                                ha='left',
                                va='center',
                                fontsize=11)
        ax_bottom.set_xticklabels(ticks, rotation=0, fontsize=12)
    else:
        ax_left.set_yticklabels([])
        ax_bottom.set_xticklabels([])

    plt.show()

    if path is not None:
        fig.savefig(path, dpi=1000, bbox_inches='tight')

