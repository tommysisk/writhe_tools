import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain
from .utils import pmf


def box_plot(values: np.ndarray,
             errors: np.ndarray = None,
             labels: np.ndarray = None,
             ylabel: str = None,
             xlabel: str = None,
             ymin: float = None,
             ymax: float = None,
             cmap: str = "viridis",
             color_list: list = None,
             title: str = None,
             ax=None,
             font_scale: float = 5,
             label_stride: int = 1,
             capsize: float = 10,
             width: float = .95,
             linewidth: float = 1,
             edgecolor: str = "black",
             alpha: float = 1,
             trunc: int = 25,
             pre_trunc: int = 25,
             rotation: float = 0):

    values, errors = [i.squeeze() if i is not None else None for i in [values, errors]]

    assert len(values.shape) == 1, \
        "Need a set of values that can be squeezed to one dimension"

    nstates = len(values)

    labels = np.arange(1, nstates + 1).astype(str) if labels is None else labels

    assert len(labels) == nstates, "Length of values and labels must be the same"

    color_list = get_color_list(nstates, cmap, trunc, pre_trunc)\
                 if color_list is None else color_list

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.bar(labels.astype(str),
           values,
           yerr=errors,
           ecolor="grey",
           color=color_list,
           capsize=10,
           width=width,
           linewidth=linewidth, edgecolor=edgecolor,
           align="center",
           alpha=alpha,
           error_kw=dict(capthick=capsize, lw=3),
           )

    ax.set_xticks(ticks=np.arange(0, nstates, label_stride), labels=labels[::label_stride])
    #     ax.set_xticklabels(labels[::label_stride], rotation=rotation)

    ax.set_xlabel(xlabel, size=6 * font_scale)
    ax.set_ylabel(ylabel, size=6 * font_scale)
    ax.set_title(title, size=6 * font_scale)

    if all(i is not None for i in (ymin, ymax)):
        ax.set_ylim(ymin, ymax)

    ax.tick_params("both", labelsize=6 * font_scale)
    ax.tick_params("x", labelrotation=rotation)
    ax.set_xlim([-.5, nstates - .5])

    return ax


def get_color_list(n_colors: int, cmap: str, trunc=0, pre_trunc=0):
    cmap = getattr(plt.cm, cmap)
    cl = [cmap(i) for i in range(cmap.N)]
    return [cl[i] for i in np.linspace(1 + pre_trunc, len(cl) - 1 - trunc, n_colors).astype(int)]


def truncate_colormap(cmap:str, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def fes2d(x: np.ndarray,
          y: np.ndarray = None,
          xlabel: str = None,
          ylabel: str = None,
          title: str = None,
          cbar: bool = True,
          cmap: str = "jet",
          vmax: float = None,
          vmin : float = None,
          cluster_centers=None,
          bins: int = 180,
          weights: np.ndarray = None,
          density: bool = False,
          extent: list = None,
          n_contours: int = 200,
          alpha_contours: float = 1,
          contour_lines: bool = False,
          alpha_lines: float = 0.6,
          scatter: bool = False,
          scatter_alpha: float = .2,
          scatter_cmap: str = "bone",
          scatter_size: float = 0.05,
          scatter_stride: int = 100,
          scatter_min: float=0.2,
          scatter_max: float=0.8,
          comp_type: str = None,
          mask: bool = True,
          font_scale: float = 1,
          cbar_shrink: float = 1,
          nxticks: int = 4,
          nyticks: int = 4,
          tick_decimals: int=2,
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
        extent = [[x.min()-extend_border, x.max()+extend_border],
                  [y.min()-extend_border, y.max()+extend_border]]

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=extent,
                                              weights=weights, density=density)

    xticks, yticks = (i[:-1] + np.diff(i) / 2 for i in (x_edges, y_edges))

    mask_index = counts <= mask_thresh

    if mask:
        counts = np.ma.masked_array(counts, mask_index)

    else:
        counts[mask_index] = counts[~mask_index].min()

    F = -np.log(counts)
    F += -F.min()

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 3))

    #ax.margins(extend_border, tight=False)

    if extend_border == 0:
        ax.set_aspect(aspect=aspect)
        #extent = None

    s = ax.contourf(F.T,
                    n_contours,
                    cmap=cmap,
                    extent=tuple(chain(*extent)), #if extent is not None and extend_border != 0 else None,
                    zorder=-1,
                    alpha=alpha_contours,
                    vmax=vmax,
                    vmin=vmin
                    )

    if contour_lines:
        ax.contour(s, colors="black", cmap=None, linewidths=1, alpha=alpha_lines)

    fmtr = lambda x, _: f"{x:.2f}"

    xlabel = f"{comp_type} 1" if (comp_type is not None) and (xlabel is None) else xlabel
    ylabel = f"{comp_type} 2" if (comp_type is not None) and (ylabel is None) else ylabel

    ax.set_title(title, size=14 * font_scale)

    ax.set_xlabel(xlabel, fontsize=14 * font_scale)
    ax.set_ylabel(ylabel, fontsize=14 * font_scale)

    # ax.set_xticks(np.linspace(nxticks, bins - nxticks, nxticks), sample_array(xticks,
    #                                                                           nxticks,
    #                                                                           figs=tick_decimals))
    # ax.set_yticks(np.linspace(nyticks, bins - nyticks, nyticks), sample_array(yticks,
    #                                                                           nyticks,
    #                                                                           figs=tick_decimals))
    # print(sample_array(yticks,
    #                                                                           nyticks,
    #                                                                           figs=tick_decimals))
    ax.tick_params(axis="x", labelsize=10 * font_scale)
    ax.tick_params(axis="y", labelsize=10 * font_scale)

    if extend_border != 0:
        ax.set_xlim(extent[0][0], extent[0][1])
        ax.set_ylim(extent[1][0], extent[1][1])


    ax.set_xticks(np.linspace(extent[0][0], extent[0][1], nxticks),
                  np.linspace(extent[0][0], extent[0][1],nxticks).round(tick_decimals))

    ax.set_yticks(np.linspace(extent[1][0], extent[1][1], nyticks),
                  np.linspace(extent[1][0], extent[1][1], nyticks).round(tick_decimals))

    # ax.set_xticklabels(sample_array(xticks,
    #                               nxticks,
    #                               figs=tick_decimals).astype(str))
    #
    # ax.set_yticklabels(sample_array(yticks,
    #                               nyticks,
    #                               figs=tick_decimals).astype(str))

    if cbar:
        cbar = plt.colorbar(s, ax=ax, format=fmtr,
                            shrink=cbar_shrink,
                            ticks=np.linspace(F.min(), F.max(), 4, endpoint=True))
        cbar.set_label("Free Energy / (kT)", size=14 * font_scale)
        cbar.ax.tick_params(labelsize=8)

    # ax.set_aspect(aspect=aspect, share=True)

    if scatter:
        c = F.flatten()[pmf([x, y], bins=bins)[2]]
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

        #ax1.autoscale_view()
        if hide_ax:
            pass
            # ax1.set_axis_off()
            # ax1.axis("off")
            # ax.set_frame_on(False)

    if cluster_centers is not None:
        # ax2 = ax.twinx().twiny()
        # ax2.set_xticks([])
        # ax2.set_yticks([])

        for j, i in enumerate(cluster_centers):
            ax.annotate(f"{j + 1}", [i[k] for k in range(2)],
                         color="black", size=str(10 * font_scale))
        if hide_ax:
            pass
            # ax2.set_axis_off()
            # ax2.axis("off")
            # ax2.set_frame_on(False)

    if hide_ax:
        ax.set_axis_off()
        ax.axis("off")
        plt.gca().set_frame_on(False)
    #ax.autoscale_view()


    return s


def get_extrema(x, extend: float = 0):
    return [x.min() - extend, x.max() + extend]


def subplots_fes2d(x: np.ndarray,
                   rows: int,
                   cols: int,
                   dscrs: list,
                   indices_list: list = None,
                   y: np.ndarray = None,
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

    x = np.stack([x, y], -1) if y is not None else x

    indices_list = list(range(len(x))) if indices_list is None else indices_list

    extent = ([get_extrema(i, extend_border) for i in x.T] if isinstance(x, np.ndarray)\
             else [get_extrema(i, extend_border) for i in np.concatenate(x)]) if extent is None and share_extent\
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

    fig.subplots_adjust(right=1.05, top=.9)
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
    fig.supylabel(ylabel)
    fig.supxlabel(xlabel)
    fig.suptitle(title, y=title_pad)
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
           tick_decimals: int=2,
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
            ticklabels = np.arange(1, nstates + 1).astype(str).tolist()\
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
                                ticks=np.linspace(c0, c1, 4, endpoint=True))
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


def subplots_proj2d(x: np.ndarray,
                    c: np.ndarray,
                    rows: int,
                    cols: int,
                    dscrs: list,
                    indices_list: list = None,
                    cmap: str = "jet",
                    dot_size: float = 0.5,
                    y: np.ndarray = None,
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
                    ):
    x = np.stack([x, y], -1) if y is not None else x

    c = c.squeeze()

    if indices_list is None:
        if x.ndim == 3:
            indices_list = list(range(len(x)))
        else:
            assert x.ndim == 2, "x must be 2 or 3 dimensional"
            indices_list = len(c) * [None]

    if c.ndim == 2:
        assert c.shape[0] == len(indices_list), "If each plot has a different coloring, number must match number of datasets"
        color_indices = list(range(len(c)))
    else:
        assert c.ndim == 1, "c must be 1 or two dimensional"
        color_indices = len(indices_list) * [None]

    extent = list(map(get_extrema, x.T)) if share_extent else None

    figsize = (2 * cols, 1.5 * rows)

    fig, axes = plt.subplots(rows, cols, sharey=sharey, sharex=sharex, figsize=figsize, constrained_layout=False)

    # if isinstance(cbar_label, str):
    #     cbar_label = len(indices_list) * [cbar_label]
    # else:
    #     assert isinstance(cbar_label, list) and (len(cbar_label) == len(indices_list))

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
                   vmin=vmin,
                   vmax=vmax,
                   cbar_shrink=1,
                   aspect=aspect)
        #ax.set_aspect(aspect)

    #fig.subplots_adjust(right=0.9, )#bottom=.2) # top=0.9
    fmtr = lambda x, _: f"{x:.3f}"
    c0, c1 = c.min(), c.max()
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
    x_offset = cols * cbar.ax.get_position().width / fig.get_size_inches()[0]
    fig.supylabel(ylabel, x= 0 - x_offset, size=(100/6) * font_scale)
    bbox = ax.get_xaxis().get_label().get_window_extent()
    fig.supxlabel(xlabel,  x = .5 - x_offset, y=bbox.y0 / (fig.bbox.height) - font_scale / 12 -.08*np.exp(1.4 - figsize[-1]), size=(100/6) * font_scale)#y=-title_pad-0.01,
    fig.suptitle(title, y=1+title_pad, x = .5 - x_offset, size=(100/6) * font_scale)
    return
