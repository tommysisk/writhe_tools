import numpy as np
import functools
import matplotlib.pyplot as plt
from .utils.filing import save_dict, load_dict
from .utils.misc import optional_import
from .plots import get_color_list

deeptime = optional_import('deeptime', 'stats' )

def reindex_msm(dtrajs: np.ndarray,
                tmats: np.ndarray = None,
                stat_dists: np.ndarray = None,
                ck_pred: np.ndarray = None,
                ck_est: np.ndarray = None,
                ck_pred_err: np.ndarray = None,
                ck_est_err: np.ndarray = None,
                obs: np.ndarray = None,
                maximize_obs=True):

    args = locals()
    results = {}
    if obs is None:
        obs = dtrajs[0]
        maximize_obs = False

    getter = lambda x, i: x[i]
    paired_outs = list(map(functools.partial(reindex_dtraj, obs=obs, maximize_obs=maximize_obs),
                           dtrajs))

    results["dtrajs"], indices = [np.stack(list(map(functools.partial(getter, i=i), paired_outs)))
                                  for i in range(2)]

    args = {key: val for key, val in args.items() if not key in "dtrajs,obs".split(",")
            and isinstance(val, (np.ndarray, list))}

    for key, val in args.items():
        # other than the stat dist,
        # the array should have square matrices in the last indices
        if isinstance(val, list):

            if len(val[0].shape) > 3:
                stack_axis = 1
            else:
                stack_axis = 0

            val = np.stack(val, axis=stack_axis)

        if val.shape[-1] != val.shape[-2]:
            # assume samples by states
            results[key] = np.take_along_axis(val, indices, axis=1)

        else:
            if len(val.shape) > 3:
                stack_axis = 1
            else:
                stack_axis = 0

            results[key] = np.stack([reindex_matrix(arr, index) for arr, index
                                     in zip(np.moveaxis(val, stack_axis, 0), indices)], stack_axis)
    return results


def reindex_dtraj(dtraj, obs, maximize_obs=True):
    """given a discrete trajectory and an observable, we reindex the trajectory
    based on the mean of the observable in each state (high to low)
    maximize_obs has been added to increase flexibility as one might want to reindex
    states in order of smallest mean value of observable"""

    states = np.sort(np.unique(dtraj)).astype(int)
    n_states = len(states)

    if n_states != (dtraj.max() + 1):
        mapping = dict(zip(states, np.arange(len(states))))

        dtraj = np.array(list(map(mapping.__getitem__, dtraj)))

    # get the sorted cluster indices based on mean of observable
    if maximize_obs:
        idx = np.array([obs[np.where(dtraj == i)[0]].mean()
                        for i in np.arange(len(states))]).argsort()[::-1]
    else:
        idx = np.array([obs[np.where(dtraj == i)[0]].mean()
                        for i in np.arange(len(states))]).argsort()

    # make a  mapping of old indices to new indices
    mapping = np.zeros(len(states))
    mapping[idx] = np.arange(len(states))

    # map the states
    new_dtraj = mapping[dtraj].astype(int)

    return new_dtraj, idx


def reindex_matrix(mat: np.ndarray, reindex: np.ndarray):
    """reindex matrix based on indices in idx"""
    # regular array
    mat = mat.squeeze()
    if len(mat.shape) == 2:
        mat = mat[reindex, :]
        mat = mat[:, reindex]

    # tmats stacked with time in the first index
    if mat.ndim == 3:
        mat = mat[:, reindex, :]
        mat = mat[:, :, reindex]

    # err tmats with low/high in first index, time in second, ...
    if mat.ndim == 4:
        mat = mat[:, :, reindex, :]
        mat = mat[:, :, :, reindex]

    return mat


def sorted_eig(x, sym=False, real=True, return_check=False):
    if sym:
        lam, v = np.linalg.eigh(x)
    else:
        lam, v = np.linalg.eig(x)

    check_real = {}
    for i, name in zip([lam, v], "eigen_vals,eigen_vecs".split(",")):
        check = np.iscomplex(i).any()
        check_real[name] = check

    if real:
        lam = np.abs(lam)

    idx = np.abs(lam).argsort()[::-1]
    lam = lam[idx]
    v = v[:, idx]
    if return_check:
        return lam, v, check_real
    else:
        return lam, v


def get_its(mats, tau: int):
    n = len(mats)
    est_lams = np.stack([sorted_eig(mat)[0] for mat in mats], axis=1)[1:]

    if (est_lams < 0).any():
        est_lams = abs(est_lams)

    predict = np.stack([-(tau * i) / np.log(est_lams[:, 0] ** i) for i in range(1, n + 1)], axis=1)
    estimate = np.stack([-(tau * i) / np.log(est_lams[:, i - 1]) for i in range(1, n + 1)], axis=1)

    return predict, estimate


def plot_its(estimate: np.ndarray, estimate_error=None, n_its: int = None,
             lag: int = 1, dt: float = .2, unit="ns", cmap: str = "jet",
             color_list: list = None,fig_width=10, fig_length=6,
             title: str = "Implied Timescales", ax=None,
             font_scale: float = 5, yscale="log"):

    """estimate: eigen vals estimated at integer multiples of the lag time
    predict: eigen vals of the initial lagtime propagated via exponentiation"""

    nprocs, nsteps = estimate.shape

    if n_its is None:
        n_its = nprocs

    color_list = get_color_list(n_colors=n_its,
                                cmap=cmap,
                                trunc=50,
                                pre_trunc=40) \
                  if color_list is None else color_list

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_length))

    lag_dt = np.arange(1, nsteps + 1) * lag * dt
    # each iteration plots a single process at all lag times
    for est_proc, color, _ in zip([estimate[i] for i in range(estimate.shape[0])],
                                  color_list,
                                  range(n_its)):
        s=ax.plot(lag_dt, est_proc, label="Estimate", color=color)
        ax.scatter(lag_dt, est_proc, color=color)
    if estimate_error is not None:
        for est_error, color, _ in zip([estimate_error[:, i] for i in range(estimate_error.shape[1])],
                                       color_list,
                                       range(n_its)):
            ax.fill_between(lag_dt,
                            est_error[0],
                            est_error[1],
                            label="Estimate",
                            color=color,
                            alpha=.2)

    ax.plot(lag_dt, lag_dt, color="black")
    ax.fill_between(lag_dt, lag_dt, color="gray", alpha=.5)
    if yscale is not None: ax.set_yscale(yscale)
    ax.set_ylabel(f"ITS ({unit})", size=5 * font_scale)
    ax.set_xlabel(rf"Lag time, $\tau$ ({unit})", size=5 * font_scale)
    ax.tick_params(axis="both", labelsize=5 * font_scale)
    ax.set_title(label=title, size=6 * font_scale)

    return s


def plot_cktest(predict: np.ndarray, estimate: np.ndarray=None,
                lag: int=1, dt: float=1, unit: str = "ns",
                predict_color: str = "red", estimate_color: str = "black",
                predict_fill_alpha=.4, estimate_fill_alpha=.4,
                predict_errors=None, estimate_errors=None,
                fill_estimate=True, title: str = None,
                figsize: tuple = (15, 15),
                font_scale: float = 1):
    """predict+errors should be of shape [2,predict/estimate.shape] where the 0th dim is upper and lower
    confidence intervals"""

    check_id = lambda x: np.all(x == np.eye(len(x)))

    # add an identity to estimate and predictions if the first matrix is not identity already
    # this is for plotting purposes

    if not check_id(predict[0]):
        print("changing pred")
        predict = np.concatenate([np.expand_dims(np.eye(predict.shape[1], predict.shape[1]), axis=0),
                                  predict])

    # make sure the predictions errors match up if they're provided

    if predict_errors is not None:
        if predict_errors.shape[1] != len(predict):
            predict_errors = np.concatenate(
                [np.expand_dims(np.stack([np.eye(predict.shape[1])] * 2), axis=1), predict_errors],
                axis=1)

    # same steps as for predictions
    if estimate is not None:
        if not check_id(estimate[0]) and len(estimate) != len(predict):
            print("changing est")
            estimate = np.concatenate([np.expand_dims(np.eye(predict.shape[1], predict.shape[1]), axis=0),
                                       estimate])

    if estimate_errors is not None:
        if estimate_errors.shape[1] != len(estimate):
            estimate_errors = np.concatenate(
                [np.expand_dims(np.stack([np.eye( predict.shape[1])] * 2), axis=1),
                 estimate_errors], axis=1)

    nsteps, nstates = predict.shape[:2]
    fig, axes = plt.subplots(nstates, nstates, figsize=figsize, sharex=True, sharey=True)
    dt_lag = np.arange(nsteps) * lag * dt
    xaxis_marker = np.linspace(0, 1, nsteps)
    padding_between = 0.2
    padding_top = 0.065

    predict_label = "Predict"
    estimate_label = "Estimate"

    for i in range(nstates):
        for j in range(nstates):
            if not predict_errors is None:
                axes[i, j].fill_between(dt_lag, predict_errors[0][:, i, j],
                                        predict_errors[1][:, i, j],
                                        color=predict_color,
                                        alpha=predict_fill_alpha)

                predict_label += "      conf. 95%"

            if estimate_errors is not None:

                if fill_estimate:
                    axes[i, j].fill_between(dt_lag[1:],
                                            estimate_errors[0][1:, i, j],
                                            estimate_errors[1][1:, i, j],
                                            color=estimate_color,
                                            alpha=estimate_fill_alpha)
                else:
                    axes[i, j].errorbar(x=dt_lag, y=estimate[:, i, j],
                                        yerr=(np.array([-1, 1]).reshape(-1, 1)
                                             * estimate_errors[:, :, i, j]
                                             + np.array([1, -1]).reshape(-1, 1)
                                             * np.stack(2 * [estimate[:, i, j]])
                                             ),
                                        color=estimate_color, alpha=1)


                estimate_label += "      conf. 95%"

            if predict is not None:
                axes[i, j].plot(dt_lag, predict[:, i, j], ls="--", color=predict_color, label=predict_label)

            if estimate is not None:
                axes[i, j].plot(dt_lag, estimate[:, i, j], color=estimate_color, label=estimate_label)

            axes[i, j].set_ylim(0, 1)
            axes[i, j].text(0.1, 0.55, str(i + 1) + ' ->' + str(j + 1),
                            transform=axes[i, j].transAxes, weight='bold', size=20 * font_scale)
            axes[i, j].set_yticks([0, .5, 1], ["0", "0.5", "1"], size=20 * font_scale)
            axes[i, j].set_xticks(dt_lag[[1, -1]], dt_lag[[1, -1]], size=20 * font_scale)

    for axi in axes.flat:
        axi.set_xlabel(None)
        axi.set_ylabel(None)

    handels, labels = axes[0, 0].get_legend_handles_labels()
    plt.subplots_adjust(top=1.0 - padding_top, wspace=padding_between, hspace=padding_between)
    fig.legend(handels, labels, ncol=7, loc="upper center", frameon=False, prop={'size': 25 * font_scale})

    if title is not None:
        fig.suptitle(title, y=0.98, x=.2, size=35 * font_scale, )

    left, bottom = fig.subplotpars.left, fig.subplotpars.bottom
    # Automatically adjust supxlabel and supylabel just before the tick labels
    shift = (15 * 0.07) / (figsize[0] / font_scale)
    fig.supxlabel(rf"Lag time, $\tau$ ({unit})", size=25 * font_scale, y=bottom - shift)
    fig.supylabel("Probability", size=25 * font_scale, x=left - shift)

    # fig.supxlabel(rf"Lag time, $\tau$ ({unit})",size=25 * font_scale,  x=0.5, y=.07, )
    # fig.supylabel("Probability",  size=25 * font_scale,x=.06, y=.5,)


    return


def plot_stat_dist(dist: np.ndarray,
                   dist_err: np.ndarray = None,
                   cmap: str = "viridis",
                   title: str = "Stationary Distribution",
                   ax=None,
                   font_scale: float = 5,
                   state_label_stride=1):
    # make the stationary distribution and it's error 1D vectors
    # (assuming abs(upper) and abs(lower) errors have been averaged):

    dist, dist_err = [i.squeeze() if i is not None else None for i in [dist, dist_err]]

    assert len(dist.shape) == 1, \
        "Need a stationary distribution that can be squeezed to one dimension"

    nstates = len(dist)
    state_labels = np.arange(1, nstates + 1)

    cmap = getattr(plt.cm, cmap)
    clist = [cmap(i) for i in range(cmap.N)]
    clist = [clist[int(i)] for i in np.linspace(10, len(clist) - 20, nstates)][::-1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.bar(state_labels, dist, yerr=dist_err,
           ecolor="grey", color=clist, capsize=10,
           width=.9, linewidth=3, edgecolor="black",
           align="center", error_kw=dict(capthick=3, lw=3),
           )

    ax.set_xticks(np.arange(1, nstates + 1, state_label_stride))
    ax.set_xticklabels(state_labels[::state_label_stride])
    ax.set_xlabel("State", size=6 * font_scale)
    ax.set_ylabel("Staionary Probability", size=6 * font_scale)
    ax.set_title(title, size=6 * font_scale)
    ax.tick_params("both", labelsize=6 * font_scale)
    ax.set_xlim(.5, nstates + .5)


def caps(string: str):
    return ''.join(map(str.capitalize, iter(string)))


def mfpt_mat(tmat, dt=1, lag=1, mu=None):
    nstates = tmat.shape[0]
    mfpt = np.zeros((nstates, nstates))
    for i in range(nstates):
        mfpt[:, i] = deeptime.markov.tools.analysis.mfpt(tmat, i, mu=mu)
    return mfpt * dt * lag


class MarkovModel:
    def __init__(self,
                 dtraj: np.ndarray = None,
                 dt: int = 1,
                 frames_cl: list = None,
                 args: dict = None):

        if args is not None:
            self.__dict__.update(args)
        else:
            assert dtraj is not None, "Must input a discrete trajectory (dtraj:::np.ndarray) if args is None"
            self.dtraj = dtraj
            self.n_states = (dtraj.max() if isinstance(dtraj, np.ndarray) else max(map(max, dtraj))) + 1
            self.dt = dt
            self.msm, self.hmm, self.pcca = [dict() for _ in range(3)]
            self.lag = None
            self.frames_cl = frames_cl

    def save(self, file: str):
        args = ["msm", "frames_cl", "pcca", "hmm", "dtraj", "lag", "dt", "n_states"]
        args = {attr: getattr(self, attr) for attr in filter(lambda x: hasattr(self, x), args)}
        save_dict(file, args)
        pass

    @classmethod
    def load(cls, file):
        return cls(args=load_dict(file))

    def estimate_msm_(self, lagtime):
        return deeptime.markov.msm.MaximumLikelihoodMSM(reversible=True).fit_fetch(self.dtraj, lagtime=lagtime)

    def estimate_msm(self, lag: int,
                     steps: int = 5):

        # base msm from which all the other models are estimated
        self.lag = lag

        self.msm.update(dict(data={}))

        self.msm["msms"] = list(map(self.estimate_msm_, np.arange(1, steps + 2) * lag))

        self.msm["data"]["tmats"] = [msm.transition_matrix
                                     for msm in self.msm["msms"]]

        self.msm["data"]["stat_dists"] = [msm.stationary_distribution
                                          for msm in self.msm["msms"]]

        self.msm["its_est"] = get_its(self.msm["data"]["tmats"], self.lag)[-1]

        observable = deeptime.markov._observables.MembershipsObservable(
            test_model=self.msm["msms"][0],
            memberships=np.eye(self.n_states)
        )

        ck = deeptime.util.validation.ck_test(models=self.msm["msms"][1:],
                                              observable=observable,
                                              test_model=self.msm["msms"][0])

        for old_key, key in zip(["predictions", "estimates"], ["ck_pred", "ck_est"]):
            self.msm["data"][key] = np.real(getattr(ck, old_key))

        return self

    def estimate_hmm(self, n_states: int):

        assert len(self.msm) != 0, "Must estimate regular msm before coarse graining (self.estimate_msm(lag=lag))"

        # make hmms
        self.hmm.update(dict(data={}))
        self.hmm["msms"] = list(map(lambda msm: msm.hmm(self.dtraj, n_states), self.msm["msms"]))

        # get the resulting discrete trajectories
        self.hmm["data"]["dtrajs"] = np.stack([hmm.metastable_assignments[self.dtraj if isinstance(self.dtraj, np.ndarray)
                                                else np.concatenate(self.dtraj)]
                                               for hmm in self.hmm["msms"]])
        # get the transition matrices
        self.hmm["data"]["tmats"] = np.stack([hmm.transition_model.transition_matrix
                                              for hmm in self.hmm["msms"]])
        # get stationary distributions
        self.hmm["data"]["stat_dists"] = np.stack([hmm.transition_model.stationary_distribution
                                                   for hmm in self.hmm["msms"]])

        # get implied timescales
        self.hmm["its_est"] = get_its(self.hmm["data"]["tmats"], self.lag)[-1]

        # get cktest data
        ck = self.hmm["msms"][0].ck_test(self.hmm["msms"][1:])

        for old_key, key in zip(["predictions", "estimates"], ["ck_pred", "ck_est"]):
            self.hmm["data"][key] = np.real(getattr(ck, old_key))

        return self

    def estimate_pcca(self, n_states: int):

        assert len(self.msm) != 0, "Must estimate regular msm before coarse graining (self.estimate_msm(lag=lag))"

        self.pcca.update(dict(data={}))
        self.pcca["msms"] = list(map(lambda msm: msm.pcca(n_states), self.msm["msms"]))

        self.pcca["data"]["dtrajs"] = np.stack([pcca.assignments[self.dtraj if isinstance(self.dtraj, np.ndarray)
                                                else np.concatenate(self.dtraj)]
                                                for pcca in self.pcca["msms"]])

        self.pcca["data"]["tmats"] = np.stack([pcca.coarse_grained_transition_matrix
                                               for pcca in self.pcca["msms"]])

        self.pcca["data"]["stat_dists"] = np.stack([pcca.coarse_grained_stationary_probability
                                                    for pcca in self.pcca["msms"]])

        self.pcca["its_est"] = get_its(self.pcca["data"]["tmats"], self.lag)[-1]

        # get cktest data
        ck = self.msm["msms"][0].ck_test(self.msm["msms"][1:],
                                         n_metastable_sets=n_states)

        for old_key, key in zip(["predictions", "estimates"], ["ck_pred", "ck_est"]):
            self.pcca["data"][key] = np.real(getattr(ck, old_key))

        return self

    # def reindex_msm(self, model_type: str, obs: np.ndarray = None, maximize_obs: bool = True):
    #
    #     assert len(getattr(self, model_type)) != 0, \
    #         "The provided model type has not been estimated yet"
    #
    #     model = getattr(self, model_type)
    #     model["data"] = reindex_msm(**model["data"], obs=obs, maximize_obs=maximize_obs)
    #     setattr(self, model_type, model)
    #
    #     return

    def its(self,
            model_type=None,
            cmap: str = "viridis",
            n_its: int = None,
            font_scale: float = 2,
            yscale="log",
            ax=None):

        assert getattr(self, model_type)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        s=plot_its(estimate=getattr(self, model_type)["its_est"],
                   n_its=n_its,
                   title=f"{caps(model_type)} Implied Timescales",
                   dt=self.dt, lag=self.lag, cmap=cmap,
                   ax=ax, font_scale=font_scale, yscale=yscale)
        return s

    def cktest(self, model_type,
               predict_color: str = "red"):

        """
        CAUTION : running this for a model with very large
        number of states will not produce a useful plot and
        may not work at all
        """

        assert len(getattr(self, model_type)) != 0, "Must estimate the chosen model type before plotting"
        data = getattr(self, model_type)["data"]

        plot_cktest(data["ck_pred"],
                    data["ck_est"],
                    lag=self.lag,
                    dt=self.dt,
                    predict_color=predict_color,
                    title=caps(model_type))
        return

    def stationary_distribution(self,
                                model_type: str = "msm",
                                cmap: str = "viridis",
                                ax=None,
                                font_scale: float = 2,
                                state_label_stride=1):

        assert len(getattr(self, model_type)) != 0, "Must estimate the chosen model type before plotting"
        data = getattr(self, model_type)["data"]

        if ax is None:
            fig, ax = plt.subplots()

        plot_stat_dist(dist=data["stat_dists"][0],
                       title=f"Stationary Distribution {caps(model_type)}",
                       cmap=cmap,
                       font_scale=font_scale,
                       ax=ax,
                       state_label_stride=state_label_stride)
