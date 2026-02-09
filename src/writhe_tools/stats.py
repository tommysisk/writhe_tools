
from .utils.indexing import (group_by,
                             sort_indices_list,
                             reindex_list,
                             product,
                             combinations)
from .utils.misc import optional_import

scipy = optional_import('scipy', 'stats' )

#from typing import List, Union, Any
from functools import partial
import warnings
import multiprocessing
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy.linalg import svd, sqrtm
import ray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
#from pyblock.blocking import reblock, find_optimal_block
import dask.array as da


def inception_distance(x, y, ax: int = 0):
    mu_x, mu_y = (mean(i, ax=ax) for i in (x, y))
    Cx, Cy = (cov(i - mu, shift=False) for i, mu in zip((x, y), (mu_x, mu_y)))
    return np.linalg.norm(mu_x - mu_y)**2 + np.trace(Cx + Cy - 2 * sqrtm(Cx @ Cy).real)

def pooled_sd(means: "1d array of trial means",
              sds: "1d array of trial sds",
              n_samples: "1d array of the number of samples used to estimate each sd and mean" = None,
              ):
    """
    For combining standard deviations.


    Can be used for combining standard deviations estimated from datasets with differing number of samples.

    If n_samples if None or a constant, then it's assumed that the number of samples is the same for all SDs and cancels out of the sum and reduces to the number of standard deviations
    being combined. As a result, this parameter can be left as None if all standard deviations are estimated using the same number of samples

    """
    if isinstance(n_samples, (float, int)) or n_samples is None:
        # in this case the number of samples cancels out
        return np.sqrt((sds ** 2 + (means - means.mean()) ** 2).sum() / len(means))
    else:
        n = n_samples.sum()
        return np.sqrt((n_samples * (sds ** 2 + (means - means.mean()) ** 2)).sum() / n)


def window_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def get_extrema(x, extend: float = 0):
    return [x.min() - extend, x.max() + extend]


def pmf1d(x: np.ndarray,
          bins: int,
          weights: np.ndarray = None,
          norm: bool = True,
          range: tuple = None,
          center_bins: bool = True):

    count, edge = np.histogram(x, bins=bins, weights=weights, range=range)
    p = count / count.sum() if norm else count
    idx = np.digitize(x, edge[1:-1])
    pi = p.flatten()[idx]
    edges = edge[:-1] + np.diff(edge) / 2 if center_bins else edge
    return p, pi, idx, edges


def pmfdd(arrays: "a list of arrays or N,d numpy array",
          bins: int,
          weights: np.ndarray = None,
          norm: bool = True,
          range: tuple = None,
          statistic: str = None,
          center_bins: bool = True):
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
    edges = (edge[:-1] + np.diff(edge) / 2 for edge in edges) if center_bins else (edge for edge in edges)
    return p, pi, idx, edges


def pmf(x: "list of arrays or array",
        bins: int,
        weights: np.ndarray = None,
        norm: bool = True,
        range: tuple = None,
        statistic: str = None,
        center_bins: bool = True):
    """
    returns : p, pi, idx, bin_centers

    """
    if isinstance(x, np.ndarray):
        x = x.squeeze()
        return pmfdd(x, bins, weights, norm, range, statistic=statistic, center_bins=center_bins) if x.ndim > 1\
                    else pmf1d(x, bins, weights, norm, range, center_bins=center_bins)
    if isinstance(x, list):
        return pmf1d(x[0], bins, weights, norm, range, center_bins=center_bins) if len(x) == 1 \
                    else pmfdd(x, bins, weights, norm, range, statistic=statistic, center_bins=center_bins)


def mean(x: np.ndarray,
         weights: np.ndarray = None,
         ax: int = 0,
         keepdims: bool = False):

    if weights is None:
        return x.mean(ax)
    else:
        shape = (1 if i != ax else x.shape[i] for i in range(len(x.shape)))
        return (weights.reshape(*shape) * x).sum(ax, keepdims=keepdims) / weights.sum()

def center(x: np.ndarray, weights: np.ndarray = None):
    return x - mean(x, weights, keepdims=True)


def std(x: np.ndarray,
        weights: np.ndarray = None,
        bessel_correction: bool = False,
        ax: int = 0):
    if weights is None:
        return x.std(ax)
    else:
        if bessel_correction:

            M = np.sum(weights) ** 2 / np.sum(weights ** 2)  # effective sample size
            N = ((M - 1) / M) * weights.sum()

        else:
            N = weights.sum()

        shape = (1 if i != ax else x.shape[i] for i in range(len(x.shape)))

    return np.sqrt(np.sum(weights.reshape(*shape) * center(x, weights) ** 2,
                          axis=ax) / N)


def standardize(x: np.ndarray,
                weights: np.ndarray = None,
                shift: bool = True,
                scale: bool = True,
                ax: int = 0):

    if scale:

        shape = (1 if i == ax else x.shape[i] for i in range(len(x.shape)))

        s = std(x, weights, ax=ax).reshape(*shape)

        return np.divide(center(x, weights) if shift else x,
                     s, out=np.zeros_like(x), where=s != 0.)

    else:
        return center(x, weights) if shift else x


def min_max_scale(x : np.ndarray, axis: int = None):
    return (x - x.min(axis=axis, keepdims=True)) / (x.max(axis=axis, keepdims=True) - x.min(axis=axis, keepdims=True))


def cov(x: np.ndarray,
        y: np.ndarray = None,
        weights: np.ndarray = None,
        norm: bool = True,
        shift: bool = True,
        scale: bool = False,
        bessel_correction: bool = False):
    n = len(x)

    if weights is not None:
        weights = weights.squeeze()
        # in case weights are a matrix in which case the norm isn't always obvious
        if weights.ndim == 1:
            if norm:
                if bessel_correction:
                    M = np.sum(weights) ** 2 / np.sum(weights ** 2)  # effective sample size
                    norm = ((M - 1) / M) * weights.sum()

                else:
                    norm = weights.sum()
            else:
                norm = 1

            apply_weights = lambda X: (weights[:, None] * X.reshape(n, -1)).reshape(n, -1)

        else:
            assert (weights.shape[0] == weights.shape[1]) and (weights.shape[1] == y.shape[0]), \
                ("weights should be a 1D matrix with len == y.shape[0]"
                 " or a square matrix with shape == (y.shape[0], y.shape[0])")
            apply_weights = lambda X: (weights @ X.reshape(n, -1)).reshape(n, -1)
            norm = 1

    else:
        norm = ((n - 1) if bessel_correction else n) if norm else 1
        apply_weights = lambda X: X

    if shift or scale:
        x, y = (standardize(i, weights=weights, shift=shift, scale=scale) if i is not None else None
                for i in (x, y))

    return (x.T @ apply_weights(y) if y is not None else x.T @ apply_weights(x)) / norm


def add_intercept(x):
    x = x.squeeze()
    if x.ndim == 1:
        return np.stack([x, np.ones_like(x)], 1)
    else:
        return np.concatenate([x, np.ones(len(x))[:, None]], 1)


def dask_svd(x,
             compressed: bool = False,
             k: int = 2,
             n_power_iter: int = 5,
             n_oversamples: int = 10,
             n_chunks: int = None,
             compute: bool = False,
             svals: bool = False,
             transposed: bool = False):
    getter = lambda x: x.compute()
    row, col = x.shape
    # this is only going to help if the svd is compressed
    if col > row:
        transposed = True
        x = x.T
    # chunking has to be in one dimension
    n_chunks = int((x.shape[0] + 1) / multiprocessing.cpu_count()) if n_chunks is None else n_chunks
    chunks = (n_chunks, x.shape[1])

    x = da.from_array(x, chunks=chunks)

    if svals:
        x = da.linalg.svd_compressed(x, k=k, n_power_iter=n_power_iter,
                                     n_oversamples=n_oversamples,
                                     compute=compute,
                                     iterator="QR")[1].compute()
        return x

    if compressed:
        x = list(map(getter, da.linalg.svd_compressed(x,
                                                      k=k,
                                                      n_power_iter=n_power_iter,
                                                      n_oversamples=n_oversamples,
                                                      compute=compute)))

    else:
        x = list(map(getter, da.linalg.svd(x)))

    if transposed:
        # to apply the transpose of the product of 3 matrices, need to flip the ordering
        return [i.T for i in x][::-1]
    else:
        return x


def svd_power(x,
                 power,
                 epsilon: float = 1e-12,
                 dask=False,
                 sym=False):
    if dask:
        u, s, vt = dask_svd(x)

    else:
        if sym:
            s, u = np.linalg.eigh(x)
            vt = u.T
        else:
            u, s, vt = svd(x,
                           full_matrices=True if epsilon is None else False,
                           lapack_driver="gesvd")

    if epsilon is not None:
        idx = s > epsilon
        s = s[idx]
        vt = vt[idx]
        u = u[:, idx]

    power = np.power(s, power)

    return u @ np.diag(power) @ vt


# beautiful linear regression
def generalized_regression(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None,
                           transform: bool = False, fit: bool = False,
                           intercept: bool = True, eps: float = 1e-10):
    if weights.squeeze().ndim != 1:
        weights = sqrtm(weights)
    # prep covariance estimator
    cov_ = functools.partial(cov, shift=False, norm=False, weights=weights)
    # add column of ones (intercept D.O.F)
    x = add_intercept(np.copy(x)) if intercept else x
    # get co-effs (solve the linear algebra problem with psuedo inv)
    u, s, vt = svd(cov_(x), full_matrices=False)
    if eps is not None:
        idx = s > 1e-10
        u, s, vt = u[:, idx], s[idx], vt[idx]

    b = vt.T @ np.diag(1 / s) @ u.T @ cov_(x, y)
    # return fit
    if transform:
        return x @ b
    # return function to transform data
    elif fit:
        return lambda x: x * b[0] + b[1]
    # return co-effs
    else:
        return b


def pca(x: np.ndarray,
        weights: np.ndarray = None,
        shift: bool = True,
        scale: bool = False,
        scale_projection: bool = False,
        n_comp: int = 10,
        dask: bool = False):
    """compute the business half of econ svd"""
    x = standardize(x, shift=shift, scale=scale, weights=weights)

    norm = np.sqrt(x.shape[0]) if scale is False and weights is None else 1

    if weights is not None:
        w = weights.astype(x.dtype, copy=True)
        w = (w / w.sum())[:, None] ** (1 / 2)
    else: w = 1.0

    s, vt = svd(x * w / norm, full_matrices=False)[1:] if not dask else\
            dask_svd(x * w / norm, k=n_comp, compressed=True)[1:]

    v = vt.T[:, :n_comp]

    projection = x @ v
    projection = (projection * w) / s[:n_comp] if scale_projection else projection
    return projection, s, v


def corr(x: np.ndarray, y: np.ndarray):
    """
    x and y should be data arrays with shape n_samples by d variables

    """
    data = np.stack([x, y], -1)
    data = data - data.mean(0, keepdims=True)
    return np.sum(np.prod(data, axis=-1), 0) / np.prod(np.linalg.norm(data, axis=0), axis=-1)


def acf(x):
    """
    Computes the autocorrelation function (ACF) of a 1D time series using FFT.

    Parameters:
    - x (np.ndarray): Input 1D signal.

    Returns:
    - acf (np.ndarray): The autocorrelation function.
    """
    N = len(x)
    # Normalize the input (subtract mean, if needed)
    x = x - np.mean(x)
    # Compute FFT of the signal
    # Compute Inverse FFT of the power spectrum
    # Normalize by variance (pearson correlation)
    return np.fft.ifft(np.abs(np.fft.fft(x, n=2*N))**2).real[:N] / (N * np.var(x))

def cross_correlation(x: np.ndarray, y: np.ndarray):
    """
    Uses the FFT to compute cross correlation between 1D time series

    x(t) * y(t+tau)

    returns: lags, correlation (automatically computes all negative to positive lags in that order)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    N = len(x)

    # Demean
    x -= x.mean()
    y -= y.mean()

    # Proper zero-padding to avoid circular wraparound
    n = 2 * N - 1

    fx = np.fft.rfft(x, n=n)
    fy = np.fft.rfft(y, n=n)
    c = np.fft.irfft(fx * np.conj(fy), n=n)

    # shift so that lag=0 is centered
    c = np.fft.fftshift(c)

    # unbiased normalization
    overlap = np.arange(1, N + 1)
    norm = np.concatenate((overlap, overlap[-2::-1]))
    c /= norm

    # divide by variance *overlap* normalization
    denom = x.std(ddof=0) * y.std(ddof=0)
    c /= denom

    # enforce zero outside valid range (suppress tiny wraparound)
    c[:N//10] = np.nan_to_num(c[:N//10])
    c[-N//10:] = np.nan_to_num(c[-N//10:])

    lags = np.arange(-N+1, N)
    return lags, c

def plot_cross_correlation(lags, corr, clip=None, normalize=True, ax=None):
    """
    Plot cross-correlation centered at lag=0.
    Automatically normalizes by the max(|corr|) within the displayed (clip) range.

    Parameters
    ----------
    lags : np.ndarray
        Lag values (output from correlation_integral_fft).
    corr : np.ndarray
        Pearson correlation coefficients.
    clip : int, optional
        Number of lag samples to display on either side of zero.
        Also determines the normalization range if normalize=True.
    normalize : bool, default=True
        Normalize correlation by max(|corr|) within the displayed range.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Creates a new one if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Matplotlib axis containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Determine symmetric clipping window around lag=0
    mid = len(lags) // 2
    if clip is not None:
        start = max(0, mid - clip)
        end   = min(len(lags), mid + clip)
        lags_window = lags[start:end]
        corr_window = corr[start:end]
    else:
        lags_window = lags
        corr_window = corr

    # Normalization: use only displayed window
    if normalize:
        ref_max = np.nanmax(np.abs(corr_window))
        if ref_max > 0:
            corr_window = corr_window / ref_max
        else:
            print("⚠️  Skipping normalization: reference max = 0")

    # --- Plot ---
    ax.plot(lags_window, corr_window, lw=1.5)
    ax.axvline(0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("Lag")
    ylabel = "Normalized correlation" if normalize else "Pearson correlation"
    ax.set_ylabel(ylabel)
    ax.set_title("Cross-correlation vs Lag")
    ax.grid(True, alpha=0.3)

    # Symmetric tick labeling
    maxlag = np.max(np.abs(lags_window))
    ax.set_xlim(-maxlag, maxlag)
    xticks = np.linspace(-maxlag, maxlag, 7, dtype=int)
    ax.set_xticks(xticks)

    plt.tight_layout()
    return ax

# plot_correlation(lags, corr, 10000, normalize=False)


def rotate_points(x: "target", y: "rotate to target", so3: bool = True):

    u, s, vt = svd(x.T @ y, full_matrices=False)
    I = np.eye(x.shape[-1])

    if so3:
        sign = np.sign(np.linalg.det(u @ vt))
        I[-1, -1] = sign

    R = u @ I @ vt

    return y @ R.T


def smooth_hist(x: np.ndarray, bins: int = 70, samples: int = 10000, norm: bool = True):
    p, edges = reindex_list(pmf(x, bins, norm=norm), [0, -1])
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
    # collect clusters into list of indices arrays SORTED BY DISTANCE FROM CENTROID
    frames_cl = sort_indices_list(indices_list=group_by(dtraj),
                                  obs=kdist,
                                  max=False)

    # return dtraj and frames for each cluster sorted by distance from centroids
    return (dtraj, frames_cl, centers, kdist, kmeans) if return_all else (dtraj, frames_cl)


def silhouette_scores(data: np.ndarray,
                      k: "int or list",
                      mult_proc: bool=False):
    """
    Silhouette score convenience function to recompute silhouette score
    with different numbers of clusters without needing to recompute pair-wise distances (if k = list).
    data: (n_samples, d_features) array of datapoints to be clustered
    k: int or list of int giving the number of clusters
    """
    distance_matrix = pairwise_distances(data,
                                         metric="euclidean",
                                         n_jobs=-1 if mult_proc else None)

    score = lambda k: silhouette_score(distance_matrix,
                                       labels=Kmeans(data, n_clusters=k, n_dim=data.shape[-1])[0],
                                       metric="precomputed").mean()
    if isinstance(k, int):
        return [k, score(k)]
    else:
        return np.stack([np.array(k), np.fromiter(map(score, k), float)], 1)


def adjust_min(x):
    idx = np.isclose(x, 0)
    if idx.sum() == 0:
        return x
    where = np.where(idx)
    x[where] = x[~idx].min()
    return x / x.sum()


def H(p, weight: bool = True):

    """ENTROPY!"""
    p = p[~np.isclose(p, 0)]
    p /= p.sum()
    return -np.sum(p * np.log2(p)) if weight else -np.sum(np.log2(p))


def mi(x,
       y,
       bins: int = None,
       weights: np.ndarray = None,
       min_count: int = None,
       shift_min: bool = False,
       norm: str = "product"):
    """
    When working with the same dataset assigned two sets of labels, x and y,
    we compute the mutual information with many normalization options.

    """

    pxy = pmf([x, y], bins=bins, weights=weights, norm=True)[0]

    if min_count is not None:
        pxy = np.where(pxy < min_count / len(x), 0, pxy)
        pxy /= pxy.sum()

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

        """Only consider non-zero bins"""

        i, j = np.where(pxy != 0)
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


def dKL(p, q, axis: "tuple or int" = None):
    # kl = p * np.log(p / q)
    # masked = np.ma.masked_array(kl, kl == np.nan)
    # p, q = common_nonzero([p, q])
    indices = np.prod(np.stack([i == 0. for i in [p, q]]), axis=0).astype(bool)
    p, q = [np.ma.masked_array(i, indices) for i in [p, q]]
    return np.sum(p * np.log(p / q), axis=axis).data


def dJS(p, q, axis: "tuple or int" = None):
    m = 0.5 * (p + q)
    return 0.5 * (dKL(p, m, axis=axis) + dKL(q, m, axis=axis))


def rmse(x, y):
    return np.sqrt(np.power(x.flatten() - y.flatten(), 2).mean())


# def block_error(x: np.ndarray):
#     """
#     x : (d, N) numpy array with d features and N measurements
#     """
#     n = x.shape[-1]
#     blocks = reblock(x)
#     optimal_indices = np.asarray(find_optimal_block(n, blocks))
#     isnan = np.isnan(optimal_indices)
#     #mode = Counter(optimal_indices[~isnan].astype(int)).most_common()[0][0]
#     optimal_indices[isnan] = -1 #mode
#     print(optimal_indices)
#     return np.asarray([blocks[i].std_err[j] for j, i in enumerate(optimal_indices.astype(int))])


def process_ids(ids):
    types = np.array(["_".join(i.split("_")[:-1]) for i in ids])
    indices_list = group_by(types)
    indices = reindex_list(indices_list, np.argsort(np.fromiter(map(np.mean, indices_list), int)))
    return indices


def lagrangian(lambdas: np.ndarray,
               constraints: np.ndarray,
               targets: np.ndarray,
               regularize: bool = False,
               sigma_reg: np.ndarray = None,
               sigma_md: np.ndarray = None):

    logits = 1 - np.dot(constraints.T, lambdas)
    shift = logits.max()
    unnormalized_weights = np.exp(logits - shift)
    norm = unnormalized_weights.sum()
    weights = unnormalized_weights / norm

    L = np.log(norm) + shift + np.dot(lambdas, targets)
    dL = targets - np.dot(constraints, weights)

    if regularize:
        L += 0.5 * np.sum(np.power(sigma_reg * lambdas, 2) + np.power(sigma_md * lambdas, 2))
        dL += np.power(sigma_reg, 2) * lambdas + np.power(sigma_md, 2) * lambdas

    return L, dL

def compute_weights(lambdas: np.ndarray, constraints: np.ndarray):
    logits = 1 - np.dot(constraints.T, lambdas)
    # Normalize exponents to avoid overflow
    weights = np.exp(logits - logits.max())
    # return weights
    return weights / np.sum(weights)

def compute_kish(weights: np.ndarray = None):
    weights /= weights.sum()
    return 100 / (len(weights) * np.power(weights, 2).sum())

def subset(indices, args):
    assert isinstance(indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
    indices = np.asarray(indices) if isinstance(indices, list) else indices
    return {i: (j[indices] if j is not None else None) for i, j in args.items()}

def get_args(constraints,
             targets,
             lambdas0,
             regularize: bool = False,
             sigma_reg: list = None,
             sigma_md: list = None,
             data_indices: list = None,
             whiten: bool = False,
             standardize: bool = False
             ):

    args = []



    if data_indices is not None:
        assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
        data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
        constraints, targets, lambdas0, sigma_reg, sigma_md = [i[data_indices] if i is not None else None for i in
                                                               [constraints, targets, lambdas0, sigma_reg, sigma_md]]

    # else:
    #     constraints, targets, lambdas0 = self.constraints, self.targets, self.lambdas0

    if whiten:
        mu = constraints.mean(1)
        w = svd_power(constraints @ constraints.T / len(constraints.T), -1 / 2)
        args.extend([((i.T - mu).reshape(-1, len(targets)) @ w).T for i in (constraints, targets)])

    elif standardize:
        mu = constraints.mean(1)
        s = constraints.std(1)
        args.extend([np.divide((i.T - mu), s, out=np.zeros_like(i.T), where=s != 0.).T
                     for i in (constraints, targets)])

    else:
        args.extend([constraints, targets])

    if regularize:
        # assert sigma_reg is not None or self.sigma_reg is not None, (
        #     "Must provide sigma_reg (regularization parameter)"
        #     "as an argument or upon instantiation")
        sigma_reg = np.zeros(len(targets)) if sigma_reg is None else sigma_reg
        sigma_md = np.zeros(len(targets)) if sigma_md is None else sigma_md
        args.extend([regularize,
                     sigma_reg,
                     sigma_md])

    else:
        args.extend([False, None, None])  # not necessary

    return dict(zip(['lambdas0', 'constraints', 'targets', 'regularize', 'sigma_reg', 'sigma_md'], [lambdas0] + args))


def reweight(constraints, targets, regularize, sigma_reg, sigma_md, lambdas0):

    result = minimize(
        lagrangian,
        lambdas0,
        method='L-BFGS-B',
        jac=True,
        args=(constraints, targets, regularize, sigma_reg, sigma_md)
    )

    weights = compute_weights(result.x, constraints)

    # if store_result:
    #     if data_indices is not None:
    #         warnings.warn("Storing parameters and weights from reweighting performed on a subset of the data.")
    #     self.lambdas = result.x
    #     self.weights = weights
    #     self.has_result = True

    weighted_averages = constraints @ weights

    return dict(lambdas=result.x,
                weights=weights,
                kish=compute_kish(weights),
                regularize=regularize,
                sigma_reg=sigma_reg,
                #data_indices=data_indices,
                #weighted_averages=weighted_averages,
                targets=targets,
                #rmse=rmse(weighted_averages, targets)
                )

def kish_scan_(constraints,
               targets,
               regularize,
               sigma_md,
               lambdas0,
               indices: np.ndarray = None,
               scale: np.array = None,
               target_kish: float = 10,
               sigma_reg_l: float = 0.001,
               sigma_reg_u: float = 20,
               steps: int = 200,
               return_scan: bool = False):


    # if data_indices is not None:

    #     assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
    #     data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
    # else:
    #     data_indices = np.arange(self.n_constraints)

    # if target_kish is not None:
    #     self.target_kish = target_kish

    if indices is not None:
        constraints, targets, sigma_md, lambdas0 = [i[indices] for i in (constraints, targets, sigma_md, lambdas0)]

    scale = np.ones_like(targets) if scale is None else scale


    kish = lambda sigma: reweight(constraints,
                                  targets,
                                  regularize,
                                  sigma,
                                  sigma_md,
                                  lambdas0)["kish"]
    reached_target = False
    sigma_optimal = sigma_reg_u * scale
    if return_scan:
        scan = []
    for sigma in np.linspace(sigma_reg_l, sigma_reg_u, steps)[::-1]:

        sigma_ = scale * sigma
        score = kish(sigma_)
        if return_scan:
            scan.append([sigma, score])

        if score < target_kish:
            reached_target = True
            break

        sigma_optimal = sigma_

    if not reached_target: print("Did not find optimal kish")
    #if store_sigma: self.sigma_reg[data_indices] = sigma_optimal

    if return_scan:
        return np.array(scan)
    else:
        return sigma_optimal



class MaxEntropyReweight():
    def __init__(self,
                 constraints: list,
                 targets: list,
                 sigma_md: list = None,
                 sigma_reg: list = None,
                 target_kish: float = 10):

        """
        constraints : list of numpy arrays each with shape (N_observations, ).
                      Each array should be paired with a target.
                      Optimization is performed to find a set of weights (N_observations)
                      that will result in a weighted average for each constraint that equals the corresponding target.

        targets : list of targets for each constraint.

        sigma_md : error of each constraint data type estimated from blocking (correlated time series data)

        sigma_reg : regularization parameter for each constraint, class method optimize_sigma_reg will find these

        target_kish : minimum kish required when searching for sigma_reg for each data type.
                      Will not necessarily match the kish of the final reweighting of all constraints combined.

        """

        self.constraints = np.asarray(constraints)
        self.targets = np.asarray(targets)
        self.lambdas0 = np.zeros(len(constraints))
        self.n_samples = len(constraints[0])
        self.n_constraints = len(self.lambdas0)

        # regularizations
        self.target_kish = target_kish

        # result status
        self.has_result = False
        self.weights = None
        self.lambdas = None

        # error in comp data
        self.sigma_md = np.asarray(constraints).std(1) if sigma_md is None else np.copy(sigma_md)

        # regularization hyperparameter (one per data type)
        self.sigma_reg = np.zeros(self.n_constraints) if sigma_reg is None else np.copy(sigma_reg)


    def compute_weights(self, lambdas, constraints: np.ndarray = None):
        constraints = self.constraints if constraints is None else constraints
        logits = 1 - np.dot(constraints.T, lambdas)
        # Normalize exponents to avoid overflow
        weights = np.exp(logits - logits.max())
        # return weights
        return weights / np.sum(weights)

    def compute_entropy(self, weights: np.ndarray = None, *args):
        if weights is None:
            assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
            weights = self.weights
        entropy = -np.sum(weights * np.log(weights + 1e-12))  # Small offset to avoid log(0)
        return entropy

    def compute_weighted_mean(self, constraints: np.ndarray = None, weights: np.ndarray = None):
        if weights is None:
            assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
            weights = self.weights
        if constraints is None:
            constraints = self.constraints
        return constraints @ weights


    def reweight(self,
                 regularize: bool = False,
                 sigma_reg: list = None,
                 data_indices: list = None,
                 store_result: bool = False,
                 whiten: bool = False,
                 standardize: bool = False
                 ):


        data_indices = np.arange(self.constraints.shape[0]) if data_indices is None else data_indices

        result = reweight(**get_args(self.constraints,
                                     self.targets,
                                     self.lambdas0,
                                     regularize,
                                     sigma_reg or self.sigma_reg,
                                     self.sigma_md,
                                     data_indices,
                                     whiten,
                                     standardize), )

        result['data_indices'] = data_indices

        constraints = self.constraints[data_indices] if data_indices is not None else self.constraints
        targets = self.targets[data_indices] if data_indices is not None else self.targets

        result['weighted_mean'] = self.compute_weighted_mean(constraints, result['weights'])

        result['rmse'] = rmse(result['weighted_mean'], targets)


        if store_result:
            if data_indices is not None:
                warnings.warn("Storing parameters and weights from reweighting performed on a subset of the data.")
            self.lambdas = result['lambdas']
            self.weights = result['weights']
            self.has_result = True


        return result

    def reset(self):
        self.weights = None
        self.lambdas = None
        self.has_result = False
        return

    def compute_kish(self, weights: np.ndarray = None):
        if weights is None:
            assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
            weights = self.weights
        return 100 / (self.n_samples * np.power(weights, 2).sum())


    def kish_scan(self,
                  data_indices: list = None,
                  target_kish: float = None,
                  sigma_reg_l: float = 0.001,
                  sigma_reg_u: float = 20,
                  steps: int = 200,
                  scale: np.array = None,
                  standardize: bool = False,
                  whiten: bool = False,
                  store_sigma: bool = False,
                  return_scan: bool = False,
                  #multi_proc: bool = False,
                  ):


        args = get_args(self.constraints,
                        self.targets,
                        self.lambdas0,
                        True,
                        self.sigma_reg,
                        self.sigma_md,
                        data_indices,
                        whiten,
                        standardize)

        args.pop('sigma_reg')

        result = kish_scan_(**args,
                            scale=scale,
                            target_kish = self.target_kish or target_kish,
                            sigma_reg_l=sigma_reg_l,
                            sigma_reg_u=sigma_reg_u,
                            steps=steps,
                            return_scan=return_scan
                            )

        if store_sigma and not return_scan:
            if data_indices is not None:
                self.sigma_reg[data_indices] = result
            else:
                self.sigma_reg = result



        return result

    def optimize_sigma_reg(self,
                           indices_list: list,
                           single_sigma_reg_l: float = 0.001,
                           single_sigma_reg_u: float = 20,
                           single_steps: int = 200,
                           global_sigma_reg_l: float = 0.01,
                           global_sigma_reg_u: float = 20,
                           global_steps: int = 60,
                           standardize: bool = False,
                           whiten: bool = False,
                           multi_proc: bool = False
                           ):

        # regularization for each data type

        args = get_args(self.constraints,
                        self.targets,
                        self.lambdas0,
                        True,
                        self.sigma_reg,
                        self.sigma_md,
                        None,
                        whiten,
                        standardize)

        args.pop('sigma_reg')

        if multi_proc:
            args = {i : ray.put(j) for i, j in args.items()}
            remote = ray.remote(kish_scan_)
            # launch

            single_regs = np.concatenate(ray.get([remote.remote(**args,
                                                                sigma_reg_l=single_sigma_reg_l,
                                                                sigma_reg_u=single_sigma_reg_u,
                                                                steps=single_steps,
                                                                indices=i
                                                                )
                                                  for i in indices_list]))

            for i in args.values():
                ray.internal.free(i)
            ray.shutdown()

            #regs = ray.get(refs)  # <-- materialize to floats

            # single_regs = np.concatenate([
            #     np.atleast_1d(reg).repeat(len(i))
            #     for reg, i in zip(result, indices_list)
            # ])

        else:
            single_regs = np.concatenate([self.kish_scan(i,
                                                         sigma_reg_l=single_sigma_reg_l,
                                                         sigma_reg_u=single_sigma_reg_u,
                                                         steps=single_steps,
                                                         )
                                          for i in indices_list])

        # global regularization - find single scalar for regularization parameters of each data type
        self.kish_scan(scale=single_regs,
                       store_sigma=True,
                       sigma_reg_l=global_sigma_reg_l,
                       sigma_reg_u=global_sigma_reg_u,
                       steps=global_steps,
                       standardize = standardize,
                       whiten =  whiten,
                       )
        return


class DensityComparator():
    """
    Estimate and compare discrete (histogram) and continuous (kernel density) of coupled datasets.

    """

    def __init__(self, data: list, weights: list = None):

        self.bounds = None
        self.data_list = data
        self.weights_list = weights
        self.kde_grid = None

    @property
    def data_list(self, array: bool = False):

        return self.data_list_

    @data_list.setter
    def data_list(self, x):

        assert isinstance(x, list), "data_list must be type list"
        assert all((isinstance(i, np.ndarray) for i in x)), "all data should be type np.ndarray"

        x = [i.squeeze() if i.squeeze().ndim > 1 else i.reshape(-1, 1) for i in x]

        assert len(set([i.shape[-1] for i in x])) == 1, "All data arrays must be the same dimension"

        self.dim = x[0].shape[-1]
        self.n_datasets = len(x)
        self.data_list_ = x
        self.set_bounds()

        return

    @property
    def weights_list(self):
        return self.weights_list_

    @weights_list.setter
    def weights_list(self, x):

        if x is not None:
            x = [i.squeeze() for i in x]
            for i, (d, w) in enumerate(zip(self.data_list, x)):
                assert len(d) == len(w), f"The number of data samples must match the number of weights : index {i}"
            self.weights_list_ = x
        else:
            self.weights_list_ = None

        return

    def set_bounds(self):
        assert self.data_list is not None, "Must have data_list_ attribute in order to estimate bounds"
        self.bounds = np.array([get_extrema(i) for i in np.concatenate(self.data_list).T])
        return

    def estimate_kde(self,
                     bins: int = 80,
                     norm: bool = True,
                     weight: bool = False,
                     bw_method=None):

        self.bins = bins

        if weight:
            assert self.weights_list is not None, "Must have weights list in order to estimate weighted KDE"

            kdes = [gaussian_kde(i.T, weights=j, bw_method=bw_method) for i, j in
                    zip(self.data_list, self.weights_list)]

        else:
            kdes = [gaussian_kde(i.T, bw_method=bw_method) for i in self.data_list]

        self.kde_grid = product(*[np.linspace(i[0], i[1], bins) for i in self.bounds]) if self.dim > 1 \
            else np.linspace(self.bounds[..., 0], self.bounds[..., 1], bins)

        setattr(self, "kdes_weighted" if weight else "kdes",
                [self.sample_kde(kde, self.kde_grid, norm=norm) for kde in kdes])

        return self

    def estimate_hist(self, bins: int = 80, norm: bool = True, weight: bool = False):

        self.bins = bins

        if weight:
            assert self.weights_list is not None, "Must have weights list in order to estimate weighted KDE"

            hists = [pmf(i, bins=bins, weights=j, norm=norm, range=self.bounds.squeeze()) for i, j in
                     zip(self.data_list, self.weights_list)]

        else:
            hists = [pmf(i, bins=bins, norm=norm, range=self.bounds.squeeze()) for i in self.data_list]

        self.hist_bin_centers = list(hists[0][-1])
        self.hist_dtrajs = [i[2] for i in hists]
        setattr(self, "hists_weighted" if weight else "hists", [i[0] for i in hists])

        return self

    @staticmethod
    def sample_kde(kde, bounds, norm: bool = True):
        sample = kde.pdf(bounds.T)
        return sample / sample.sum() if norm else sample

    @property
    def n_datasets(self):
        return self.n_datasets_

    @n_datasets.setter
    def n_datasets(self, x):
        assert isinstance(x, int), "Number of datasets should be an integer"
        self.n_datasets_ = x
        self.data_pairs = combinations(np.arange(x)).astype(int)
        return

    @staticmethod
    def cos_similarity(x, y, axis: "tuple of ints or int" = None):
        return np.sum(x * y, axis=axis) / np.sqrt(np.sum(x ** 2, axis=axis) * np.sum(y ** 2, axis=axis))

    @property
    def bins(self):
        return self.bins_

    @bins.setter
    def bins(self, x: int):
        if hasattr(self, "bins_"):
            if self.bins_ != x:
                warnings.warn(
                    f"Bins to use in densitiy estimators has already been set to {self.bins_}."
                    f" Changing to {x}. Consider recomputing all densities")
        self.bins_ = x

        return

    def compare(self, attr: str, weight: bool = False, metric: callable = None,
                pairs: np.ndarray = None, weight0: bool = None, weight1: bool = None,
                iterate: bool = False):

        pairs = self.data_pairs if pairs is None else pairs

        assert attr in ("hists", "kdes"), "Density to compare must be either 'kdes' or 'hists' regardless of weighting"

        if "hists" in attr:
            warnings.warn((
                "Using densities defined by histograms in the computation of a comparison metric"
                " can cause counter intuitive results because empty bins are masked out to prevent nans")
            )

        if all(i is None for i in (weight0, weight1)):

            attr = attr + "_weighted" if weight else attr
            assert hasattr(self, attr), f"Class must have {attr} in order to compare"
            density = getattr(self, attr)
            d0, d1 = np.stack([density[i] for i in pairs[:, 0]]), np.stack([density[i] for i in pairs[:, 1]])

        else:

            assert all(i is not None for i in (weight0, weight1)), \
                "Must specify weighting for both datasets if weighting is specified for either"

            densities = []
            for i in (weight0, weight1):

                attr_ = attr + "_weighted" if i else attr
                assert hasattr(self, attr_), f"Class must have {attr_} in order to compare"
                densities.append(getattr(self, attr_))

            d0, d1 = [np.stack([d[i] for i in p]) for p, d in zip(pairs.T, densities)]

        if iterate:
            assert metric is not None, 'Metric cannot be left to None if iterate is True'
            return np.array([metric(i, j) for i, j in zip(d0, d1)])

        metric = partial(self.cos_similarity if metric is None else metric, axis=(1, 2) if d0.ndim > 2 else -1)

        return metric(d0, d1)

    def plot_kde(self,
                 weight: bool = False,
                 title: str = None,
                 dscrs: list = None,
                 dscr: str = None,
                 figsize: tuple = (6, 1.8),
                 xlabel: str = None,
                 kwargs: dict = {}):

        attr = 'kdes_weighted' if weight else 'kdes'
        assert hasattr(self, attr), "Must estimate KDEs before plotting"

        if dscrs is not None:
            assert len(dscrs) == self.n_datasets, "Number of labels must match the number of datasets"
        else:
            dscrs = self.n_datasets * [""]

        title = ("Weighted Kernel Densities" if weight else "Kernel Densities") if title is None else title

        if dscr is not None:
            title = f"{title} : {dscr}"

        density = getattr(self, "kdes_weighted" if weight else "kdes")

        if self.dim == 2:
            from .plots import subplots_proj2d
            args = dict(figsize=figsize, sharex=True, sharey=True, cbar_label="Density")
            args.update(kwargs)

            subplots_proj2d(self.kde_grid, c=np.stack(density),
                            rows=1, cols=self.n_datasets,
                            dscrs=dscrs, title=title,
                            xlabel=xlabel,
                            **args)
        elif self.dim == 1:
            fig, axes = plt.subplots(1,
                                     self.n_datasets,
                                     figsize=(self.n_datasets, 1) if figsize is None else figsize,
                                     constrained_layout=True,
                                     sharey=True)

            for i, ax, label in zip(density, axes.flat, dscrs):
                ax.plot(self.kde_grid, i)
                ax.set_title(label)

            fig.supylabel("Density")
            fig.supxlabel(xlabel)
            fig.suptitle(title)

            return

        else:
            raise Exception("Currently, data must be 1 or 2 dimensional to plot")

        return

    def plot_hist(self,
                  weight: bool = False,
                  title: str = None,
                  dscrs: list = None,
                  dscr: str = None,
                  kwargs: dict = {}):

        if weight:
            assert self.weights_list is not None, "Must provide weights for weighted histogram plot"

        if dscrs is not None:
            assert len(dscrs) == self.n_datasets, "Number of labels must match the number of datasets"
        else:
            dscrs = self.n_datasets * [""]

        # weights = self.weights_list if self.weights_list is not None and weight else self.n_datasets * [None]

        title = ("Weighted Histogram Densities" if weight else "Histogram Densities") if title is None else title

        if dscr is not None:
            title = f"{title} : {dscr}"

        args = dict(figsize=(6, 1.8), title_pad=1.11, sharex=True, sharey=True)
        args.update(kwargs)

        from .plots import subplots_fes2d

        subplots_fes2d(x=self.data_list,
                       cols=self.n_datasets,
                       title=f"Reweighted : {title}" if weight else title,
                       dscrs=dscrs,
                       weights_list=self.weights_list if weight else None,
                       rows=1,
                       extent=self.bounds,
                       **args)
        #TODO make an option to plot 1D histogram data
        return



#
#
# def conditional_ray(attr):
#     """
#     conditional ray decorator
#     """
#     def decorator(func):
#         def inner(*args, **kwargs):
#             is_ray = getattr(args[0], attr)
#             return ray.remote(func) if is_ray else func
#         return inner
#     return decorator
#
#
# class MaxEntropyReweight():
#     def __init__(self,
#                  constraints: list,
#                  targets: list,
#                  sigma_md: list = None,
#                  sigma_reg: list = None,
#                  target_kish: float = 10):
#
#         """
#         constraints : list of numpy arrays each with shape (N_observations, ).
#                       Each array should be paired with a target.
#                       Optimization is performed to find a set of weights (N_observations)
#                       that will result in a weighted average for each constraint that equals the corresponding target.
#
#         targets : list of targets for each constraint.
#
#         sigma_md : error of each constraint data type estimated from blocking (correlated time series data)
#
#         sigma_reg : regularization parameter for each constraint, class method optimize_sigma_reg will find these
#
#         target_kish : minimum kish required when searching for sigma_reg for each data type.
#                       Will not necessarily match the kish of the final reweighting of all constraints combined.
#
#         """
#
#         self.constraints = np.asarray(constraints)
#         self.targets = np.asarray(targets)
#         self.lambdas0 = np.zeros(len(constraints))
#         self.n_samples = len(constraints[0])
#         self.n_constraints = len(self.lambdas0)
#
#         # regularizations
#         self.target_kish = target_kish
#
#         # result status
#         self.has_result = False
#         self.weights = None
#         self.lambdas = None
#
#         # error in comp data
#         self.sigma_md = np.asarray(constraints).std(1) if sigma_md is None else np.copy(sigma_md)
#
#         # regularization hyperparameter (one per data type)
#         self.sigma_reg = np.zeros(self.n_constraints) if sigma_reg is None else np.copy(sigma_reg)
#
#         self.is_ray = False
#
#     def compute_weights(self, lambdas, constraints: np.ndarray = None):
#         constraints = self.constraints if constraints is None else constraints
#         logits = 1 - np.dot(constraints.T, lambdas)
#         # Normalize exponents to avoid overflow
#         weights = np.exp(logits - logits.max())
#         # return weights
#         return weights / np.sum(weights)
#
#     def compute_entropy(self, weights: np.ndarray = None, *args):
#         if weights is None:
#             assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
#             weights = self.weights
#         entropy = -np.sum(weights * np.log(weights + 1e-12))  # Small offset to avoid log(0)
#         return entropy
#
#     def compute_weighted_mean(self, weights: np.ndarray = None):
#         if weights is None:
#             assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
#             weights = self.weights
#         return self.constraints @ weights
#
#     def lagrangian(self,
#                    lambdas,
#                    constraints: np.ndarray,
#                    targets: np.ndarray,
#                    regularize: bool = False,
#                    sigma_reg: np.ndarray = None,
#                    sigma_md: np.ndarray = None):
#
#         logits = 1 - np.dot(constraints.T, lambdas)
#         shift = logits.max()
#         unnormalized_weights = np.exp(logits - shift)
#         norm = unnormalized_weights.sum()
#         weights = unnormalized_weights / norm
#
#         L = np.log(norm / self.n_samples) + shift - 1 + np.dot(lambdas, targets)
#         dL = targets - np.dot(constraints, weights)
#
#         if regularize:
#             L += 0.5 * np.sum(np.power(sigma_reg * lambdas, 2) + np.power(sigma_md * lambdas, 2))
#             dL += np.power(sigma_reg, 2) * lambdas + np.power(sigma_md, 2) * lambdas
#
#         return L, dL
#
#     def reweight(self,
#                  regularize: bool = False,
#                  sigma_reg: list = None,
#                  data_indices: list = None,
#                  store_result: bool = False,
#                  whiten: bool = False,
#                  standardize: bool = False
#                  ):
#
#         args = []
#
#         if data_indices is not None:
#             assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
#             data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
#             constraints, targets, lambdas0 = [getattr(self, i)[data_indices] for i in
#                                               ["constraints", "targets", "lambdas0"]]
#
#         else:
#             constraints, targets, lambdas0 = self.constraints, self.targets, self.lambdas0
#
#         if whiten:
#             mu = constraints.mean(1)
#             w = svd_power(constraints @ constraints.T / len(constraints.T), -1 / 2)
#             args.extend([((i.T - mu).reshape(-1, len(targets)) @ w).T for i in (constraints, targets)])
#
#         elif standardize:
#             mu = constraints.mean(1)
#             s = std(constraints, ax=1)
#             args.extend([np.divide((i.T - mu), s, out=np.zeros_like(i.T), where=s != 0.).T
#                          for i in (constraints, targets)])
#
#         else:
#             args.extend([constraints, targets])
#
#         if regularize:
#             assert sigma_reg is not None or self.sigma_reg is not None, (
#                 "Must provide sigma_reg (regularization parameter)"
#                 "as an argument or upon instantiation")
#             args.extend([regularize,
#                          np.asarray(sigma_reg) if sigma_reg is not None else self.sigma_reg[data_indices].squeeze(),
#                          self.sigma_md[data_indices].squeeze()])
#
#         else:
#             args.extend([False, None, None])  # not necessary
#
#         result = minimize(
#             self.lagrangian,
#             lambdas0,
#             method='L-BFGS-B',
#             jac=True,
#             args=tuple(args)
#         )
#
#         weights = self.compute_weights(result.x, constraints)
#
#         if store_result:
#             if data_indices is not None:
#                 warnings.warn("Storing parameters and weights from reweighting performed on a subset of the data.")
#             self.lambdas = result.x
#             self.weights = weights
#             self.has_result = True
#
#         weighted_averages = constraints @ weights
#
#         return dict(lambdas=result.x,
#                     weights=weights,
#                     kish=self.compute_kish(weights),
#                     regularize=args[-2],
#                     sigma_reg=args[-1],
#                     data_indices=data_indices,
#                     weighted_averages=weighted_averages,
#                     targets=targets,
#                     rmse=rmse(weighted_averages, targets)
#                     )
#
#     def reset(self):
#         self.weights = None
#         self.lambdas = None
#         self.has_result = False
#         return
#
#     def compute_kish(self, weights: np.ndarray = None):
#         if weights is None:
#             assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
#             weights = self.weights
#         return 100 / (self.n_samples * np.power(weights, 2).sum())
#
#     @conditional_ray("is_ray")
#     def kish_scan_(self,
#                    data_indices: list = None,
#                    target_kish: float = None,
#                    sigma_reg_l: float = 0.001,
#                    sigma_reg_u: float = 20,
#                    steps: int = 200,
#                    scale: np.array = 1.,
#                    standardize: bool = False,
#                    whiten: bool = False,
#                    store_sigma: bool = False,
#                    return_scan: bool = False):
#
#         if data_indices is not None:
#
#             assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
#             data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
#         else:
#             data_indices = np.arange(self.n_constraints)
#
#         if target_kish is not None:
#             self.target_kish = target_kish
#
#         kish = lambda sigma: self.reweight(regularize=True,
#                                            sigma_reg=sigma,
#                                            data_indices=data_indices,
#                                            store_result=False,
#                                            standardize=standardize,
#                                            whiten=whiten)["kish"]
#         reached_target = False
#         sigma_optimal = sigma_reg_u * scale
#         if return_scan:
#             scan = []
#         for sigma in np.linspace(sigma_reg_l, sigma_reg_u, steps)[::-1]:
#
#             sigma_ = scale * sigma
#             score = kish(sigma_)
#             if return_scan:
#                 scan.append([sigma, score])
#
#             if score < self.target_kish:
#                 reached_target = True
#                 break
#
#             sigma_optimal = sigma_
#
#         if not reached_target: print("Did not find optimal kish")
#         if store_sigma: self.sigma_reg[data_indices] = sigma_optimal
#
#         if return_scan:
#             return np.array(scan)
#         else:
#             return sigma_optimal
#
#     def kish_scan(self,
#                   data_indices: list = None,
#                   target_kish: float = None,
#                   sigma_reg_l: float = 0.001,
#                   sigma_reg_u: float = 20,
#                   steps: int = 200,
#                   scale: np.array = 1,
#                   standardize: bool = False,
#                   whiten: bool = False,
#                   store_sigma: bool = False,
#                   multi_proc: bool = False):
#
#         args = dict(self=self,
#                     data_indices=data_indices,
#                     target_kish=target_kish,
#                     sigma_reg_l=sigma_reg_l,
#                     sigma_reg_u=sigma_reg_u,
#                     steps=steps,
#                     scale=scale,
#                     store_sigma=store_sigma,
#                     standardize = standardize,
#                     whiten =  whiten,
#                     )
#
#         if multi_proc:
#             self.is_ray = True
#             return_ = self.kish_scan_().remote(**args)
#             self.is_ray = False
#             return return_
#         else:
#             return self.kish_scan_()(**args)
#
#     def optimize_sigma_reg(self,
#                            indices_list: list,
#                            single_sigma_reg_l: float = 0.001,
#                            single_sigma_reg_u: float = 20,
#                            single_steps: int = 200,
#                            global_sigma_reg_l: float = 0.01,
#                            global_sigma_reg_u: float = 20,
#                            global_steps: int = 60,
#                            standardize: bool = False,
#                            whiten: bool = False,
#                            multi_proc: bool = False
#                            ):
#
#         # regularization for each data type
#
#         if multi_proc:
#             # launch
#             refs = [self.kish_scan(i,
#                                    sigma_reg_l=single_sigma_reg_l,
#                                    sigma_reg_u=single_sigma_reg_u,
#                                    steps=single_steps,
#                                    multi_proc=True,
#                                    standardize=standardize,
#                                    whiten=whiten)
#                     for i in indices_list]
#
#             regs = ray.get(refs)  # <-- materialize to floats
#
#             single_regs = np.concatenate([
#                 np.atleast_1d(reg).repeat(len(i))
#                 for reg, i in zip(regs, indices_list)
#             ])
#
#         else:
#             single_regs = np.concatenate([
#                 np.atleast_1d(self.kish_scan(i,
#                                              sigma_reg_l=single_sigma_reg_l,
#                                              sigma_reg_u=single_sigma_reg_u,
#                                              steps=single_steps,
#                                              multi_proc=False,
#                                              standardize=standardize,
#                                              whiten=whiten)
#                               ).repeat(len(i))
#                 for i in indices_list
#             ])
#
#         # global regularization - find single scalar for regularization parameters of each data type
#         self.kish_scan(scale=single_regs,
#                        store_sigma=True,
#                        sigma_reg_l=global_sigma_reg_l,
#                        sigma_reg_u=global_sigma_reg_u,
#                        steps=global_steps,
#                        standardize = standardize,
#                        whiten =  whiten,
#                        )
#         return

