import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import ray
from sklearn.cluster import KMeans
from pyblock.blocking import reblock, find_optimal_block
from collections import Counter
from .plots import subplots_fes2d, subplots_proj2d
from functools import partial
import warnings
from scipy.stats import gaussian_kde
from .utils import (group_by,
                    sort_indices_list,
                    reindex_list,
                    product,
                    combinations,
                    pmf,
                    get_extrema)


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

    pxy = pmf([x, y], bins=bins, weights=weights, norm=True)[0]

    if min_count is not None:
        pxy = np.where(pxy < min_count, 0, pxy)
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

        """Only consider non-zero bins. Renormalizes distributions after removing zeros.
           This gives same result as SKLearn but we can factor in weights using cluster
           similarity function"""

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


def block_error(x: np.ndarray):
    """
    x : (d, N) numpy array with d features and N measurements
    """
    n = x.shape[-1]
    blocks = reblock(x)
    optimal_indices = np.asarray(find_optimal_block(n, blocks))
    isnan = np.isnan(optimal_indices)
    mode = Counter(optimal_indices[~isnan].astype(int)).most_common()[0][0]
    optimal_indices[isnan] = mode
    return np.asarray([blocks[i].std_err[j] for j, i in enumerate(optimal_indices.astype(int))])


def process_ids(ids):
    types = np.array(["_".join(i.split("_")[:-1]) for i in ids])
    indices_list = group_by(types)
    indices = reindex_list(indices_list, np.argsort(np.fromiter(map(np.mean, indices_list), int)))
    return indices


def conditional_ray(attr):
    """conditional ray decorator
    """

    def decorator(func):

        def inner(*args, **kwargs):

            is_ray = getattr(args[0], attr)

            if is_ray:
                return ray.remote(func)
            else:
                return func

        return inner

    return decorator


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
        self.sigma_md = block_error(np.asarray(constraints)) if sigma_md is None else np.copy(sigma_md)

        # regularization hyperparameter (one per data type)
        self.sigma_reg = np.zeros(self.n_constraints) if sigma_reg is None else np.copy(sigma_reg)

        self.is_ray = False

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

    def compute_weighted_mean(self, weights: np.ndarray = None):
        if weights is None:
            assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
            weights = self.weights
        return self.constraints @ weights

    def lagrangian(self,
                   lambdas,
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

        L = np.log(norm / self.n_samples) + shift - 1 + np.dot(lambdas, targets)
        dL = targets - np.dot(constraints, weights)

        if regularize:
            L += 0.5 * np.sum(np.power(sigma_reg * lambdas, 2) + np.power(sigma_md * lambdas, 2))
            dL += np.power(sigma_reg, 2) * lambdas + np.power(sigma_md, 2) * lambdas

        return L, dL

    def reweight(self,
                 regularize: bool = False,
                 sigma_reg: list = None,
                 data_indices: list = None,
                 store_result: bool = False
                 ):

        args = []

        if data_indices is not None:
            assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
            data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
            constraints, targets, lambdas0 = [getattr(self, i)[data_indices] for i in
                                              ["constraints", "targets", "lambdas0"]]

        else:
            constraints, targets, lambdas0 = self.constraints, self.targets, self.lambdas0

        args.extend([constraints, targets])

        if regularize:
            assert sigma_reg is not None or self.sigma_reg is not None, (
                "Must provide sigma_reg (regularization parameter)"
                "as an argument or upon instantiation")
            args.extend([regularize,
                         np.asarray(sigma_reg) if sigma_reg is not None else self.sigma_reg[data_indices].squeeze(),
                         self.sigma_md[data_indices].squeeze()])

        else:
            args.extend([False, None, None])  # not necessary

        result = minimize(
            self.lagrangian,
            lambdas0,
            method='L-BFGS-B',
            jac=True,
            args=tuple(args)
        )

        weights = self.compute_weights(result.x, constraints)

        if store_result:
            if data_indices is not None:
                warnings.warn("Storing parameters and weights from reweighting performed on a subset of the data.")
            self.lambdas = result.x
            self.weights = weights
            self.has_result = True

        weighted_averages = constraints @ weights

        return dict(lambdas=result.x,
                    weights=weights,
                    kish=self.compute_kish(weights),
                    regularize=args[-2],
                    sigma_reg=args[-1],
                    data_indices=data_indices,
                    weighted_averages=weighted_averages,
                    targets=targets,
                    rmse=rmse(weighted_averages, targets)
                    )

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

    @conditional_ray("is_ray")
    def kish_scan_(self,
                   data_indices: list = None,
                   target_kish: float = None,
                   sigma_reg_l: float = 0.001,
                   sigma_reg_u: float = 20,
                   steps: int = 200,
                   scale: np.array = 1,
                   store_sigma: bool = False):

        if data_indices is not None:

            assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
            data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
        else:
            data_indices = np.arange(self.n_constraints)

        if target_kish is not None:
            self.target_kish = target_kish

        kish = lambda sigma: self.reweight(regularize=True,
                                           sigma_reg=sigma,
                                           data_indices=data_indices,
                                           store_result=False)["kish"]
        reached_target = False
        sigma_optimal = sigma_reg_u * scale
        for sigma in np.linspace(sigma_reg_l, sigma_reg_u, steps)[::-1]:

            sigma = scale * sigma

            if kish(sigma) < self.target_kish:
                reached_target = True
                break

            sigma_optimal = sigma

        if not reached_target: print("Did not find optimal kish")
        if store_sigma: self.sigma_reg[data_indices] = sigma_optimal

        return sigma_optimal

    def kish_scan(self,
                  data_indices: list = None,
                  target_kish: float = None,
                  sigma_reg_l: float = 0.001,
                  sigma_reg_u: float = 20,
                  steps: int = 200,
                  scale: np.array = 1,
                  store_sigma: bool = False,
                  multi_proc: bool = False):

        args = dict(self=self,
                    data_indices=data_indices,
                    target_kish=target_kish,
                    sigma_reg_l=sigma_reg_l,
                    sigma_reg_u=sigma_reg_u,
                    steps=steps,
                    scale=scale,
                    store_sigma=store_sigma
                    )

        if multi_proc:
            self.is_ray = True
            return_ = self.kish_scan_().remote(**args)
            self.is_ray = False
            return return_
        else:
            return self.kish_scan_()(**args)

    def optimize_sigma_reg(self,
                           indices_list: list,
                           single_sigma_reg_l: float = 0.001,
                           single_sigma_reg_u: float = 20,
                           single_steps: int = 200,
                           global_sigma_reg_l: float = 0.01,
                           global_sigma_reg_u: float = 20,
                           global_steps: int = 60,
                           multi_proc: bool = False
                           ):

        # regularization for each data type

        single_regs = np.concatenate([self.kish_scan(i,
                                                     sigma_reg_l=single_sigma_reg_l,
                                                     sigma_reg_u=single_sigma_reg_u,
                                                     steps=single_steps,
                                                     multi_proc=multi_proc) * np.ones(len(i))
                                      for i in indices_list])

        # global regularization - find single scalar for regularization parameters of each data type
        self.kish_scan(scale=single_regs,
                       store_sigma=True,
                       sigma_reg_l=global_sigma_reg_l,
                       sigma_reg_u=global_sigma_reg_u,
                       steps=global_steps,
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

            kdes = [gaussian_kde(i.T, weights=j, bw_method=bw_method) for i, j in zip(self.data_list, self.weights_list)]

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

        self.hist_bin_centers = [i[-1] for i in hists]
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
                    f"Bins to use in densitiy estimators has already been set to {self.bins_}. Changing to {x}. Consider recomputing all densities")
        self.bins_ = x

        return

    def compare(self, attr: str, weight: bool = False, metric: callable = None,
                pairs: np.ndarray = None, weight0: bool = None, weight1: bool = None):

        pairs = self.data_pairs if pairs is None else pairs

        assert attr in ("hists", "kdes"), "Density to compare must be either 'kdes' or 'hists' regardless of weighting"

        if "hists" in attr:
            warnings.warn((
                              "Using densities defined by histograms in the computation of a comparision metric can cause counter intuitive results "
                              "because empty bins are masked out to prevent nans")
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

        metric = partial(self.cos_similarity if metric is None else metric, axis=(1, 2) if d0.ndim > 2 else -1)

        return metric(d0, d1)

    def plot_kde(self,
                 weight: bool = False,
                 title: str = None,
                 dscrs: list = None,
                 dscr: str = None,
                 figsize: tuple=(6, 1.8),
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

            #fig.tight_layout()

            pass

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

        #weights = self.weights_list if self.weights_list is not None and weight else self.n_datasets * [None]

        title = ("Weighted Histogram Densities" if weight else "Histogram Densities") if title is None else title

        if dscr is not None:
            title = f"{title} : {dscr}"

        args = dict(figsize=(6, 1.8), title_pad=1.11, sharex=True, sharey=True)
        args.update(kwargs)

        subplots_fes2d(x=self.data_list,
                       cols=self.n_datasets,
                       title=f"Reweighted : {title}" if weight else title,
                       dscrs=dscrs,
                       weights_list=self.weights_list if weight else None,
                       rows=1,
                       extent=self.bounds,
                       **args)
        return

