
import multiprocessing
import os
import functools
import numpy as np
from deeptime.numeric import spd_inv_split
from scipy.linalg import svd
import dask.array as da


def rotate_points(x: "target", y: "rotate to target"):
    u, s, vt = svd(x.T @ y, full_matrices=False)

    sign = np.sign(np.linalg.det(vt.T @ u.T))

    I = np.eye(x.shape[-1])

    if x.shape[-1] >= 3:
        I[-1, -1] = sign

    R = u @ I @ vt

    return y @ R.T


def mean(x: np.ndarray, weights: np.ndarray = None, ax: int = 0):
    return x.mean(ax) if weights is None else (weights[:, None] * x).sum(ax) / weights.sum()


def center(x: np.ndarray, weights: np.ndarray = None):
    return x - mean(x, weights)


def std(x: np.ndarray,
        weights: np.ndarray = None,
        bessel_correction: bool = False,
        ax: int = 0):
    if weights is None:
        return x.std(ax)
    else:
        if bessel_correction:

            M = np.sum(weights)**2 / np.sum(weights**2) # effective sample size
            N = ((M - 1) / M) * weights.sum()

        else:
            N = weights.sum()

        mu = mean(x, weights, ax=ax)
        return np.sqrt(np.sum(weights[:, None] * (x - mu)**2, ax) / N)


def standardize(x: np.ndarray,
                weights: np.ndarray = None,
                shift: bool = True, scale: bool = True, ax: int = 0):
    mu = mean(x, weights, ax) if shift else 0
    s = std(x, weights, ax) if scale else 1
    return np.divide((x - mu), s, out=np.zeros_like(x), where = s!= 0.)


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


class CCA:

    def __init__(self,
                 x0: np.ndarray,
                 x1: np.ndarray,
                 dim: int = None,
                 epsilon: float = 1e-10,
                 ):

        self.x0 = x0
        self.x1 = x1
        self.x0_mean, self.x1_mean = x0.mean(0), x1.mean(0)
        self.epsilon = epsilon
        self.dim = dim if dim is not None else self.x0.shape[-1]
        self.has_fit = False

    def fit(self,
            x0: np.ndarray = None,
            x1: np.ndarray = None,
            dim: int = None,
            epsilon: float = None,
            dask: bool = False):

        if dim is None:
            dim = self.dim

        else:
            self.dim = dim

        if epsilon is None:
            epsilon = self.epsilon

        if x0 is None:
            x0 = self.x0

        if x1 is None:
            x1 = self.x1

        assert all(i is not None for i in (x0, x1)), \
            ("x0 and x1 must not be None in order to fit. "
             "provide x0 and x1 as arguments to this function or the init method")

        # currently can't use dask to get speed up
        pseudo_half_inv = {attr: spd_inv_split(cov(x - getattr(self, f"{attr}_mean"), shift=False),
                                               epsilon=epsilon)
                           for attr, x in zip("x0,x1".split(","), [x0, x1])}

        cca = pseudo_half_inv["x0"].T @ cov(x0, x1) @ pseudo_half_inv["x1"]

        if dask:
            v0, svals, v1_t = dask_svd(cca, compressed=True, k=self.dim)
        else:
            v0, svals, v1_t = svd(cca, full_matrices=False, lapack_driver='gesvd')

        v1 = v1_t.T

        v0, v1 = [v[..., :dim] for v in [v0, v1]]
        svals = svals[:dim]

        for key in "pseudo_half_inv,cca,v0,svals,v1,dim,epsilon".split(","):
            setattr(self, key, locals()[key])

        self.has_fit = True

        return self

    def transform(self,
                  x: "a numpy array related to the data used in fit or str(x0,x1)" = None,
                  dim: int = None,
                  scale: bool = False):

        assert self.has_fit, "Must fit before transforming (use fit_transform or fit)"

        if dim is not None:
            assert dim <= self.dim, "Cannot transform onto dimension larger than fit estimate"
            self.dim = dim
        else:
            dim = self.dim

        if x is None:

            vecs = [getattr(self, f"v{i}")[:, :dim] for i in range(2)]

            if scale:
                vecs = [v * self.svals[:dim] for v in vecs]

            return [getattr(self, xi) - getattr(self, f"{xi}_mean") @ self.pseudo_half_inv[xi] @ v
                    for xi, v in zip(["x0", "x1"], vecs)]

        elif isinstance(x, str):

            assert x in "x0,x1".split(","), "if x is str, input should be str(x0 or x1)"

            v = getattr(self, f"v{x[-1]}")[:dim]

            if scale:
                v = v * self.svals[:dim]

            return (getattr(self, x) - getattr(self, f"{x}_mean")) @ self.pseudo_half_inv[x] @ v

        else:

            assert (isinstance(x, np.ndarray) and
                    (x.shape[1] == self.pseudo_half_inv["x0"].shape[0])), \
                "must input a numpy array of the appropriate shape"

            v = self.v0[:, :dim]

            if scale:
                v = v * self.svals[:dim]

            return (x - self.x0_mean) @ self.pseudo_half_inv["x0"] @ v


class tCCA(CCA):
    def __init__(self, data: np.ndarray, lag: int, dim: int = None, epsilon: float = 1e-10):
        super().__init__(x0=data[:-lag], x1=data[lag:], dim=dim, epsilon=epsilon)
        self.lag = lag
        self.data = data

    def transform(self,
                  x: "a numpy array related to the data used in fit or str(x0,x1)" = None,
                  dim: int = None,
                  scale=False):

        # transform all the data onto the forward singular functions (extrapolation)
        if x is None:
            return super().transform(x=self.data, dim=dim, scale=scale)
        else:
            return super().transform(x=x, dim=dim, scale=scale)

    def fit_transform(self, x0: np.ndarray = None, x1: np.ndarray = None,
                      x: str = None, dim: int = None, scale: bool = False,
                      dask=False):

        self.fit(x0=x0, x1=x1, dim=dim, dask=dask)

        return self.transform(x=x, dim=dim, scale=scale)


def _tcca_score(data: np.ndarray,
                lag: int,
                dim: int = 10,
                dscr: str = None,
                path: str = None,
                project: bool = True,
                singular_vectors: bool = False):

    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        path = os.getcwd()

    dscr = f"{dscr}_" if dscr is not None else ""
    file = f"{path}/{dscr}tCCA_lag_{lag}"

    tcca = tCCA(np.double(data), lag=lag, dim=dim).fit()
    np.save(f"{file}_singular_values", tcca.svals)

    if singular_vectors:
        np.save(f"{file}_singular_vectors", tcca.v0)

    if project:
        projection = tcca.transform()
        np.save(f"{file}_projection", projection)

    return


def _tcca_scores(data: np.ndarray,
                 lags: np.ndarray,
                 dim: int = 20,
                 dscr: str = None,
                 path: str = None,
                 project: bool = True,
                 singular_vectors: bool = False):

    func = functools.partial(_tcca_score,
                             data=data,
                             dscr=dscr,
                             path=path,
                             dim=dim,
                             project=project,
                             singular_vectors=singular_vectors)

    for lag in lags: func(lag=lag)

    return


def add_intercept(x):
    x = x.squeeze()
    if x.ndim == 1:
        return np.stack([x, np.ones_like(x)], 1)
    else:
        return np.concatenate([x, np.ones(len(x))[:, None]], 1)


def matrix_power(x,
                 power,
                 epsilon: float = 1e-12,
                 dask=False, sym=True):
    if dask:
        u, s, vt = dask_svd(x)

    else:
        if sym:
            s, u = np.linalg.eigh(x)
            vt = u.T
        else:
            u, s, vt = svd(x, full_matrices=False, lapack_driver="gesvd")

    if epsilon is not None:
        idx = s > epsilon
        s = s[idx]
        vt = vt[idx]
        u = u[:, idx]

    power = np.power(s, power)

    return u @ np.diag(power) @ vt


# beautiful linear regression
def generalized_regression(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None,
                           transform: bool = False, fit: bool = False, intercept: bool = True):
    # prep covariance estimator
    cov_ = functools.partial(cov, shift=False, norm=False, weights=weights)

    # add column of ones (intercept D.O.F)
    x = add_intercept(np.copy(x)) if intercept else x

    # get co-effs (solve the linear algebra problem with psuedo inv)
    b = matrix_power(cov_(x), -1, sym=True) @ cov_(x, y)

    # return fit
    if transform:
        return x @ b

    # return function to transform data
    elif fit:
        return lambda x: x * b[0] + b[1]

    # return co-effs
    else:
        return b


def rotate_points(x: "target", y: "rotate to target"):
    u, s, vt = svd(x.T @ y, full_matrices=False)

    sign = np.sign(np.linalg.det(vt.T @ u.T))

    I = np.eye(x.shape[-1])

    if x.shape[-1] >= 3:
        I[-1, -1] = sign

    R = u @ I @ vt

    return y @ R.T
