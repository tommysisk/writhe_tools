
from .utils.misc import optional_import
deeptime = optional_import('deeptime', 'stats' )
from .stats import cov, dask_svd

import os
import functools
import numpy as np
from deeptime.numeric import spd_inv_split
from scipy.linalg import svd




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

        self.matrix = cca

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
        else:
            dim = self.dim

        if x is None:

            vecs = [getattr(self, f"v{i}")[:, :dim] for i in range(2)]

            if scale:
                vecs = [v * self.svals[:dim] for v in vecs]

            return [(getattr(self, xi) - getattr(self, f"{xi}_mean")) @ self.pseudo_half_inv[xi] @ v
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


