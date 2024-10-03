import numpy as np
import scipy
from scipy import interpolate
from sklearn.cluster import KMeans
from .utils import group_by, sort_indices_list


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


    if range is not None:
        idx = np.stack([np.digitize(value, edge[1:-1]) for edge, value in zip(edges, arrays.T)]) + 1

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


def smooth_hist(x, bins=70, samples=10000):
    p, _, _, edges = pmf(x, bins, norm=True)
    f = interpolate.interp1d(edges, p, kind="cubic")
    x = np.linspace(edges[0], edges[-1], samples)
    return x, f(x)


def Kmeans(p: np.ndarray,
           n_clusters: int,
           n_dim: int,
           n_init: int = 10,
           max_iter: int = 300,
           init: str = "k-means++",
           return_all: bool = False):

    """
    full return: dtraj, frames_cl, centers, kdist, kmeans
    """

    p = np.copy(p[..., :n_dim])
    # use kmeans class from sklearn
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                    max_iter=max_iter, init=init)
    # fit the clustering, return labels
    dtraj = kmeans.fit_predict(p)
    # get distance from center for each frame
    kdist = kmeans.transform(p).min(1)
    # get cluster centers
    centers = kmeans.cluster_centers_
    # collect clusters into list of indices arrays
    frames_cl = sort_indices_list(indices_list=group_by(dtraj),
                                  obs=kdist,
                                  max=False)

    # return dtraj and frames for each cluster sorted by distance from centroids
    return (dtraj, frames_cl, centers, kdist, kmeans) if return_all else (dtraj, frames_cl)


def adjust_min(x):
    idx = np.where(x == 0)
    x[idx] = x[x != 0].min()
    return x / x.sum()


def H(p, weight: bool = True):
    """ENTROPY!"""
    p = p[p != 0]
    p /= p.sum()
    return -np.sum(p * np.log2(p)) if weight else -np.sum(np.log2(p))


def mi(x,
       y,
       bins: int = 50,
       weights: np.ndarray = None,
       min_count: int = None,
       shift_min: bool = False,
       norm: str = "product"):

    """
    When working with the same dataset assigned two sets of labels, x and y,
    we compute the mutual information with many normalization options.

    """

    pxy = pmf([x, y], bins=bins, weights=weights, norm=True)

    if shift_min:

        """add small amount to probabilities with zero weight.
           This avoids the need to remove bins from the distributions.
           Weights are renormalized after addition.
           As a result, we can do our math in matrix form :)
            """

        pxy = adjust_min(pxy)

        px, py = pxy.sum(1), pxy.sum(0)

        info = pxy * np.log2(np.diag(1 / px) @ pxy @ np.diag(1 / py))

    else:

        """Only consider non-zero bins. Renormalizes distributions after removing zeros.
           This gives same result as SKLearn but we can factor in weights using cluster
           similarity function"""

        i, j = np.where(pxy != 0) if min_count is None else np.where(pxy >= min_count)

        px, py = pxy.sum(1), pxy.sum(0)

        pij = pxy[i, j]

        info = pij * np.log2(pij / (px[i] * py[j]))

    norm = 2 * len(pxy) if norm == "state" \
        else np.log2(len(x)) if norm == "sample" \
        else (H(px) + H(py)) / 2 if norm == "sum" \
        else np.sqrt(H(px)) * np.sqrt(H(py)) if norm == "product" \
        else max(H(px), H(py)) if norm == "max" \
        else min(H(px), H(py)) if norm == "min" \
        else H(pxy, weight=True) if norm == "joint" \
        else 1

    return info / norm
