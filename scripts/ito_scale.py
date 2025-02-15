#!/usr/bin/env python


# region Imports


import numpy as np
import torch
from pytorch_lightning import Trainer
from torch import nn
from tqdm import tqdm
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
import math
import copy
import warnings
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger as Logger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import contextlib
from datetime import timedelta
from writhe_tools.utils.graph_utils import GraphDataSet
from writhe_tools.writhe_nn import WritheMessage, AddWritheEdges
import argparse
import os


# endregion


# region Noise and Sampling
class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),
                                  self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                               torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


class SampleTracker:
    def __init__(self, should_track=True):
        self.should_track = should_track
        self.traj = []

    def add_frame(self, frame):
        if self.should_track:
            self.traj.append(frame)

    def get_traj(self):
        if self.should_track:
            return torch.stack(self.traj, dim=0)
        else:
            return


def model_wrapper(
        model,
        noise_schedule,
        model_type="noise",
        model_kwargs={},
        guidance_type="uncond",
        condition=None,
        unconditional_condition=None,
        guidance_scale=1.,
        classifier_fn=None,
        classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).

        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            ``
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).


    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="dpmsolver++",
            correcting_x0_fn=None,
            correcting_xt_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995,
    ):
        """Construct a DPM-Solver.

        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

        We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
        can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
        dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
        DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
        DPMs (such as stable-diffusion).

        To support advanced algorithms in image-to-image applications, we also support corrector functions for
        both x0 and xt.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
            correcting_x0_fn: A `str` or a function with the following format:
                ```
                def correcting_x0_fn(x0, t):
                    x0_new = ...
                    return x0_new
                ```
                This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                ```
                x0_pred = data_pred_model(xt, t)
                if correcting_x0_fn is not None:
                    x0_pred = correcting_x0_fn(x0_pred, t)
                xt_1 = update(x0_pred, xt, t)
                ```
                If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
            correcting_xt_fn: A function with the following format:
                ```
                def correcting_xt_fn(xt, t, step):
                    x_new = ...
                    return x_new
                ```
                This function is to correct the intermediate samples xt at each sampling step. e.g.,
                ```
                xt = ...
                xt = correcting_xt_fn(xt, t, step)
                ```
            thresholding_max_val: A `float`. The max value for thresholding.
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
            Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
            with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3, ] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3, ] * (K - 1) + [1]
            else:
                orders = [3, ] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2, ] * K
            else:
                K = steps // 2 + 1
                orders = [2, ] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1, ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(torch.tensor([0, ] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                    sigma_t / sigma_s * x
                    - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False,
                                            solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(
            s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s) * x
                    - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                        torch.exp(log_alpha_t - log_alpha_s) * x
                        - (sigma_t * phi_1) * model_s
                        - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                        torch.exp(log_alpha_t - log_alpha_s) * x
                        - (sigma_t * phi_1) * model_s
                        - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1. / 3., r2=2. / 3., model_s=None, model_s1=None,
                                           return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(
            s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                        (sigma_s1 / sigma_s) * x
                        - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                    (sigma_s2 / sigma_s) * x
                    - (alpha_s2 * phi_12) * model_s
                    + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                        (sigma_t / sigma_s) * x
                        - (alpha_t * phi_1) * model_s
                        + (alpha_t * phi_2) * D1
                        - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                        (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                        - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                    (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                    - (sigma_s2 * phi_12) * model_s
                    - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                        (torch.exp(log_alpha_t - log_alpha_s)) * x
                        - (sigma_t * phi_1) * model_s
                        - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                        (torch.exp(log_alpha_t - log_alpha_s)) * x
                        - (sigma_t * phi_1) * model_s
                        - (sigma_t * phi_2) * D1
                        - (sigma_t * phi_3) * D2
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(
            t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = (
                        (sigma_t / sigma_prev_0) * x
                        - (alpha_t * phi_1) * model_prev_0
                        - 0.5 * (alpha_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                        (sigma_t / sigma_prev_0) * x
                        - (alpha_t * phi_1) * model_prev_0
                        + (alpha_t * (phi_1 / h + 1.)) * D1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = (
                        (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                        - (sigma_t * phi_1) * model_prev_0
                        - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                        (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                        - (sigma_t * phi_1) * model_prev_0
                        - (sigma_t * (phi_1 / h - 1.)) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(
            t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
            )
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None,
                                     r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate,
                                                            solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate,
                                                           solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5,
                            solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1,
                                                                                               solver_type=solver_type,
                                                                                               **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1,
                                                                                    return_intermediate=True,
                                                                                    solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2,
                                                                                              solver_type=solver_type,
                                                                                              **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise.

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
                method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
                atol=0.0078, rtol=0.05, return_intermediate=False,
                ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
                           method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero,
                           solver_type=solver_type,
                           atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
               method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
               atol=0.0078, rtol=0.05, return_intermediate=False,
               ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver.
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep',
                              'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep',
                              'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol,
                                             solver_type=solver_type)
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step,
                                                         solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in tqdm(range(order, steps + 1), leave=False, desc='ODE Sample'):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order,
                                                         solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps,
                                                                                                  order=order,
                                                                                                  skip_type=skip_type,
                                                                                                  t_T=t_T, t_0=t_0,
                                                                                                  device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order, ] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order,
                                                          device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x


#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


# endregion


# region Exponential Moving Average


class ExponentialMovingAverage(torch.nn.Module):
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
            Note that EMA is computed on *all* provided parameters,
            regardless of whether or not they have `requires_grad = True`;
            this allows a single EMA object to be consistantly used even
            if which parameters are trainable changes step to step.

            If you want to some parameters in the EMA, do not pass them
            to the object in the first place. For example:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

            will ignore parameters that do not require grad.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
            self,
            parameters: torch.nn.Parameter,
            decay: float,
            use_num_updates: bool = True
    ):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        # self.register_buffer("shawdow_params", [p.clone().detach() for p in parameters])
        self.shadow_params = torch.nn.ParameterList([
            p.clone().detach()
            for p in parameters
        ])
        self.collected_params = None
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [p for p in parameters]
        self._synced_device = False

    def _get_parameters(
            self,
            parameters: torch.nn.Parameter,
    ) -> torch.nn.Parameter:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
            self,
            parameters: torch.nn.Parameter = None
    ) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        #        if not self._synced_device:
        #            target_device = parameters[0].device
        #            self.shadow_params = [_.to(target_device) for _ in self.shadow_params]
        #            self._synced_device = True
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = (s_param - param)
                # tmp will be a new tensor so we can do in-place
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(
            self,
            parameters: torch.nn.Parameter = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(
            self,
            parameters: torch.nn.Parameter = None
    ) -> None:
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
        ]

    def restore(
            self,
            parameters: torch.nn.Parameter = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
            self,
            parameters: torch.nn.Parameter = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            store()
            copy_to()
            try:
                ...
            finally:
                restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )


# endregion


# region Beta Scheduler
class NoiseSchedulerBase:
    def __init__(self, diffusion_steps, beta_min=None, beta_max=None):
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_alphas(self):
        return 1 - self.get_betas()

    def get_betas(self):
        raise NotImplementedError


class AlphaBarScheduler(NoiseSchedulerBase):
    def get_betas(self):
        alpha_bars = self.get_alpha_bars()
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

        return betas

    def get_alpha_bars(self):
        raise NotImplementedError


class BetaScheduler(NoiseSchedulerBase):
    def get_alpha_bars(self):
        betas = self.get_betas()
        alpha_bars = []
        alpha_bar = 1
        for beta in betas:
            alpha_bars.append(alpha_bar)
            alpha = 1 - beta
            alpha_bar = alpha * alpha_bar

        return torch.Tensor(alpha_bars)

    def get_betas(self):
        raise NotImplementedError


class LinearBetaScheduler(BetaScheduler):
    def get_betas(self):
        betas = [
            self.beta_min + (t / self.diffusion_steps) * (self.beta_max - self.beta_min)
            for t in range(self.diffusion_steps)
        ]
        return torch.Tensor(betas)


class SigmoidalBetaScheduler(BetaScheduler):
    def get_betas(self):
        ts = torch.linspace(-8, -4, self.diffusion_steps)
        betas = torch.sigmoid(ts)
        return betas


class CosineScheduler(AlphaBarScheduler):
    def get_alpha_bars(self):
        s = 0.008
        nu = 1

        import matplotlib.pyplot as plt

        t = torch.linspace(0, self.diffusion_steps, self.diffusion_steps + 1)
        arg = ((t / self.diffusion_steps + s) ** nu) / (1 + s) * torch.pi / 2 / torch.pi

        return torch.cos(arg) ** 2


class SigmoidalScheduler(AlphaBarScheduler):
    def get_alpha_bars(self):
        t = torch.linspace(-5, 5, self.diffusion_steps + 1)
        alpha_bars = torch.sigmoid(-t)
        return alpha_bars


class PolynomialScheduler(AlphaBarScheduler):
    def __init__(self, diffusion_steps, power=2):
        super().__init__(diffusion_steps)
        self.power = power

    def get_alpha_bars(self):
        s = 1e-4
        steps = self.diffusion_steps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, self.power)) ** 2

        alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

        precision = 1 - 2 * s

        alphas2 = precision * alphas2 + s

        return torch.tensor(alphas2, dtype=torch.float32)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


# pylint: disable=import-outside-toplevel, unused-variable
def main_plot():
    import matplotlib.pyplot as plt

    s500 = PolynomialScheduler(1000, 3)
    s1000 = SigmoidalBetaScheduler(1000)

    alpha_bars500 = s500.get_alpha_bars()
    alpha_bars1000 = s1000.get_alpha_bars()

    import numpy as np

    plt.plot(np.linspace(0, 1, len(alpha_bars500)), alpha_bars500, label="500")
    plt.plot(np.linspace(0, 1, len(alpha_bars1000)), alpha_bars1000, label="1000")

    plt.legend()
    plt.show()


# endregion

# region Embedding
class MLP(torch.nn.Module):
    def __init__(self, f_in, f_hidden, f_out, skip_connection=False):
        super().__init__()
        self.skip_connection = skip_connection

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(f_in, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_hidden),
            torch.nn.LayerNorm(f_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(f_hidden, f_out),
        )

    def forward(self, x):
        if self.skip_connection:
            return x + self.mlp(x)

        return self.mlp(x)


class AddEdgeIndex(torch.nn.Module):
    def __init__(self, n_neighbors=None, cutoff=None):
        super().__init__()
        self.n_neighbors = n_neighbors if n_neighbors else 10000
        self.cutoff = cutoff if cutoff is not None else float("inf")

    def forward(self, batch):
        batch = batch.clone()
        edge_index = self.generate_edge_index(batch)
        batch.edge_index = edge_index.to(batch.x.device)
        return batch


class AddSpatialEdgeFeatures(torch.nn.Module):
    def forward(self, batch, *_, **__):
        r = batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]

        edge_dist = r.norm(dim=-1)
        edge_dir = r / (1 + edge_dist.unsqueeze(-1))

        batch.edge_dist = edge_dist
        batch.edge_dir = edge_dir
        return batch


class InvariantFeatures(torch.nn.Module):
    """
    Implement embedding in child class
    All features that will be embedded should be in the batch
    """

    def __init__(self, feature_name, type_="node"):
        super().__init__()
        self.feature_name = feature_name
        self.type = type_

    def forward(self, batch):
        embedded_features = self.embedding(batch[self.feature_name])

        name = f"invariant_{self.type}_features"
        if hasattr(batch, name):
            batch[name] = torch.cat([batch[name], embedded_features], dim=-1)
        else:
            batch[name] = embedded_features

        return batch


class NominalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, n_types, feature_type="node"):
        super().__init__(feature_name, feature_type)
        self.embedding = torch.nn.Embedding(n_types, n_features)


class DeviceTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("device_tracker", torch.tensor(1))

    @property
    def device(self):
        return self.device_tracker.device


class PositionalEncoder(DeviceTracker):
    def __init__(self, dim, length=10):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.length = length
        self.max_rank = dim // 2

    def forward(self, x):
        encodings = [self.positional_encoding(x, rank) for rank in range(self.max_rank)]
        return torch.cat(
            encodings,
            axis=1,
        )

    def positional_encoding(self, x, rank):
        sin = torch.sin(x / self.length * rank * np.pi)
        cos = torch.cos(x / self.length * rank * np.pi)
        assert (
                cos.device == self.device
        ), f"batch device {cos.device} != model device {self.device}"
        return torch.stack((cos, sin), axis=1)


class PositionalEmbedding(InvariantFeatures):
    def __init__(self, feature_name, n_features, length):
        super().__init__(feature_name)
        assert n_features % 2 == 0, "n_features must be even"
        self.rank = n_features // 2
        self.embedding = PositionalEncoder(n_features, length)


class CombineInvariantFeatures(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.mlp = MLP(n_features_in, n_features_out, n_features_out)

    def forward(self, batch):
        batch.invariant_node_features = self.mlp(batch.invariant_node_features)
        return batch


class AddEquivariantFeatures(DeviceTracker):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def forward(self, batch):
        eq_features = torch.zeros(
            batch.batch.shape[0],
            self.n_features,
            3,
        )
        batch.equivariant_node_features = eq_features.to(self.device)
        return batch


# endregion

# region DDPM
class DDPMBase(pl.LightningModule):
    def __init__(
            self,
            diffusion_steps=1000,
            lr=1e-3,
            noise_schedule="sigmoid",
            alpha_bar_weight=True,
            dont_evaluate=True,
            batch_size: int = 128,
            n_features=32,
            n_node_types: int = 20,
            n_bond_types: int = 210,
            score_layers=3,
            diff_steps=1000,
            dist_encoding="positional_encoding",
            writhe_layer: bool = True,
            cross_product: bool = False,
            n_atoms: int = 20,
    ):

        super().__init__()
        self.score_model = PaiNNScore(batch_size=batch_size,
                                      n_features=n_features,
                                      n_node_types=n_node_types,
                                      n_bond_types=n_bond_types,
                                      score_layers=score_layers,
                                      diff_steps=diff_steps,
                                      dist_encoding=dist_encoding,
                                      writhe_layer=writhe_layer,
                                      cross_product=cross_product,
                                      n_atoms=n_atoms)
        self.save_hyperparameters()
        self.diffusion_steps = diffusion_steps
        self.alpha_bar_weight = alpha_bar_weight
        self.dont_evaluate = dont_evaluate
        self.fails = 0

        if noise_schedule == "sigmoid":
            self.beta_scheduler = SigmoidalBetaScheduler(diffusion_steps)
        if noise_schedule == "cosine":
            self.beta_scheduler = CosineScheduler(diffusion_steps)
        if noise_schedule.startswith("polynomial"):
            split = noise_schedule.split("_")
            if len(split) == 1:
                power = 2
            else:
                power = float(split[1])

            self.beta_scheduler = PolynomialScheduler(
                diffusion_steps, float(power)
            )

        self.register_buffer("betas", self.beta_scheduler.get_betas())
        self.register_buffer("alphas", self.beta_scheduler.get_alphas())
        self.register_buffer("alpha_bars", self.beta_scheduler.get_alpha_bars())

        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), decay=0.99
        )
        self.lr = lr
        self.last_evaluation = 10

    def on_after_backward(self):
        for _, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(
                        "detected inf or nan values in gradients. not updating model parameters",
                        flush=True,
                    )
                    self.fails += 1
                    if self.fails > 1000:
                        raise ValueError("Too many fails, stopping training")

                    self.zero_grad()
                    break

    def forward(self, batch):
        return self.score_model(batch)

    def training_step(self, batch, _):
        global_step = self.trainer.global_step

        loss, batch = self.get_loss(batch, return_batch=True)

        loss = loss.mean()

        if torch.isnan(loss):
            print(f"Loss is NaN at global_step {global_step}", flush=True)
            self.fails += 1
            return loss
            #  raise ValueError(f"Loss is NaN at global_step {global_step}")

        if self.should_evaluate(global_step):
            try:
                self.evaluate(global_step)
                self.last_evaluation = global_step
            except:  # pylint: disable=bare-except
                pass

        self.log("loss", loss, prog_bar=True)

        return loss

    def should_evaluate(self, global_step):
        if global_step % 50000 == 0 and global_step != 0:
            return True

        return False

    def save_checkpoint(self, path):
        trainer = Trainer()
        trainer.strategy.connect(self)
        trainer.save_checkpoint(path)

    def on_before_zero_grad(self, *args, **kwargs):  # pylint: disable=unused-argument
        self.ema.update(self.score_model.parameters())

    def prepare_batch(self, batch):
        ts_diff = torch.randint(
            1, self.diffusion_steps, [len(batch), 1], device=self.device
        )

        batch["alpha_bars"] = self.alpha_bars[ts_diff]
        batch["ts_diff"] = ts_diff

        batch["epsilon"] = self.get_epsilon(batch)
        batch["corr"] = self.get_corrupted(batch)

        return batch

    def get_loss(self, batch, return_batch=False):

        batch = self.prepare_batch(batch)
        epsilon_hat = self.forward(batch)
        loss = self.calculate_loss(epsilon_hat, batch)

        if self.alpha_bar_weight:
            loss *= self.alpha_bars[batch["ts_diff"]].squeeze()

        if return_batch:
            return loss, batch
        return loss

    def get_epsilon(self, batch):
        raise NotImplementedError

    def get_corrupted(self, batch):
        raise NotImplementedError

    def calculate_loss(self, epsilon_hat, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class GeometricDDPM(DDPMBase):
    def get_epsilon(self, batch):
        epsilon = batch.clone()
        epsilon.x = torch.randn(batch.x.shape, device=self.device)
        return epsilon

    def get_epsilon_like(self, conf):
        epsilon = conf.clone()
        epsilon.x = torch.randn(conf.x.shape, device=self.device)
        return epsilon

    def get_corrupted(self, batch):
        alpha_bars = batch["alpha_bars"][batch.batch]
        corrupted = batch.clone()

        corrupted.x = (
                torch.sqrt(alpha_bars) * batch.x
                + torch.sqrt(1 - alpha_bars) * batch["epsilon"].x
        )
        return corrupted

    def calculate_loss(self, epsilon_hat, batch):
        loss = nn.functional.mse_loss(
            epsilon_hat.x, batch["epsilon"].x, reduction="none"
        ).sum(-1)
        return scatter(loss, batch.batch, reduce="mean")

    def sample_like(self, conf, ode_steps=0, save_traj=False):
        corr = self.get_epsilon_like(conf)
        batch = {"corr": corr}
        return self.sample(batch, ode_steps, save_traj=save_traj)

    def sample_cond(self, cond, lag=1, ode_steps=0):
        corr = self.get_epsilon_like(cond)
        lag = torch.ones(len(cond), device=self.device) * lag
        batch = {"corr": corr, "cond": cond, "lag": lag}
        return self.sample(batch, ode_steps)

    def sample(self, batch, ode_steps=0, save_traj=False):
        if ode_steps:
            return self._ode_sample(batch, ode_steps=ode_steps, save_sample_traj=save_traj)
        return self._sample(batch)

    def _sample(self, batch):
        with torch.no_grad():
            for t_diff in tqdm(
                    torch.arange(self.diffusion_steps - 1, 0, -1, dtype=torch.int64)
            ):
                ts_diff = (
                        torch.ones(
                            [len(batch["corr"].batch), 1],
                            device=self.device,
                            dtype=torch.int32,
                        )
                        * t_diff
                )

                batch["ts_diff"] = ts_diff
                batch = self.denoise_sample(batch)
        return batch["corr"]

    def denoise_sample(self, batch):  # , t, x, epsilon_hat):
        epsilon_hat = self.forward(batch)
        epsilon = self.get_epsilon_like(batch["corr"])

        preepsilon_scale = (self.alphas[batch["ts_diff"]]) ** (-0.5)
        epsilon_scale = (1 - self.alphas[batch["ts_diff"]]) / (
                1 - self.alpha_bars[batch["ts_diff"]]
        ) ** 0.5
        post_sigma = ((self.betas[batch["ts_diff"]]) ** 0.5) * epsilon.x
        batch["corr"].x = (
                preepsilon_scale * (batch["corr"].x - epsilon_scale * epsilon_hat.x)
                + post_sigma
        )

        return batch

    def _ode_sample(self, batch, ode_steps=100, save_sample_traj=False):

        sample_tracker = SampleTracker(save_sample_traj)

        noise_schedule = NoiseScheduleVP(
            "discrete",
            betas=self.betas,
        )

        def t_diff_and_forward(x, t):
            t = t[0]
            batch["ts_diff"] = (
                    torch.ones_like(batch["corr"].batch, device=self.device) * t
            )
            batch["corr"].x = x
            sample_tracker.add_frame(x)
            epsilon_hat = self.forward(batch)
            return epsilon_hat.x

        wrapped_model = model_wrapper(t_diff_and_forward, noise_schedule)
        dpm_solver = DPM_Solver(wrapped_model, noise_schedule)

        batch["corr"].x = dpm_solver.sample(batch["corr"].x, ode_steps)
        if save_sample_traj:
            return batch["corr"], sample_tracker.get_traj()
        return batch["corr"]


# endregion

# region PAINN
class PaiNNBase(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
            self,
            batch_size: int = 128,
            n_features=64,
            n_layers=5,
            n_features_out=1,
            length_scale=10,
            dist_encoding="positional_encoding",
            use_edge_features=True,
            writhe_layer: bool = True,
            cross_product: bool = False,
            n_atoms: int = 20
    ):

        super().__init__()
        message = SE3Message if cross_product else Message
        layers = []
        for _ in range(n_layers):
            layers.append(
                message(
                    n_features=n_features,
                    length_scale=length_scale,
                    dist_encoding=dist_encoding,
                    use_edge_features=use_edge_features,
                )

            )
            if writhe_layer:
                # if _ == n_layers - 1:
                #     layers.append(WritheMessage(n_atoms=n_atoms, n_features=n_features, batch_size=batch_size))

                layers.append(WritheMessage(n_atoms=n_atoms,
                                            n_features=n_features,
                                            batch_size=batch_size,
                                            segment_length=1,
                                            gaussian_bins=False,
                                            #bin_std=bin_std
                                            )
                              )

            layers.append(Update(n_features))

        layers.append(Readout(n_features, n_features_out))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        return self.layers(batch)


class Message(torch.nn.Module):
    def __init__(
            self,
            n_features=64,
            length_scale=10,
            dist_encoding="positional_encoding",
            use_edge_features=True,
            n_atoms=20,
    ):
        super().__init__()
        self.n_features = n_features
        self.use_edge_features = use_edge_features

        assert dist_encoding in (
            a := ["positional_encoding", "soft_one_hot"]
        ), f"positional_encoder must be one of {a}"

        if dist_encoding in ["positional_encoding", None]:
            self.positional_encoder = PositionalEncoder(
                n_features, length=length_scale
            )

        phi_in_features = 2 * n_features if use_edge_features else n_features
        self.phi = MLP(phi_in_features, n_features, 4 * n_features)
        self.w = MLP(n_features, n_features, 4 * n_features)
        self.u = MLP(n_features, n_features, 4 * n_features)

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        in_features = batch.invariant_node_features[src_node]

        if self.use_edge_features:
            in_features = torch.cat(
                [in_features, batch.invariant_edge_features], dim=-1
            )

        positional_encoding = self.positional_encoder(batch.edge_dist)
        wr = torch.stack([torch.sin(batch.writhe / 10 * i * torch.pi)
                          for i in torch.arange(self.n_features)],
                            dim=-1)

        gates, scale_edge_dir, ds, de = torch.split(
            self.phi(in_features) * self.w(positional_encoding) * self.u(wr),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(
            gates, batch.equivariant_node_features[src_node]
        )

        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dv = scaled_edge_dir + gated_features
        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features += dv
        batch.invariant_node_features += ds
        batch.invariant_edge_features += de

        return batch


class SE3Message(nn.Module):
    def __init__(
            self,
            n_features=64,
            length_scale=10,
            dist_encoding="positional_encoding",
            use_edge_features: bool = True,
    ):
        super().__init__()
        self.n_features = n_features

        self.positional_encoder = PositionalEncoder(n_features, length=length_scale)

        phi_in_features = 2 * n_features
        self.phi = MLP(phi_in_features, n_features, 5 * n_features)
        self.w = MLP(n_features, n_features, 5 * n_features)

    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        in_features = torch.cat(
            [
                batch.invariant_node_features[src_node],
                batch.invariant_edge_features,
            ],
            dim=-1,
        )

        positional_encoding = self.positional_encoder(batch.edge_dist)

        gates, scale_edge_dir, ds, de, cross_product_gates = torch.split(
            self.phi(in_features) * self.w(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(gates, batch.equivariant_node_features[src_node])
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        src_equivariant_node_features = batch.equivariant_node_features[dst_node]
        cross_produts = torch.cross(dst_node_edges, src_equivariant_node_features, dim=-1)

        gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

        dv = scaled_edge_dir + gated_features + gated_cross_products
        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features = batch.equivariant_node_features + dv
        batch.invariant_node_features = batch.invariant_node_features + ds
        batch.invariant_edge_features = batch.invariant_edge_features + de

        return batch


class CrossMessage(torch.nn.Module):
    def __init__(self,
                 n_features=64,
                 length_scale=10,
                 dist_encoding="positional_encoding",
                 use_edge_features: bool = False,
                 n_atoms=20

                 ):
        super().__init__()
        self.n_features = n_features
        self.use_edge_features = use_edge_features

        assert dist_encoding in (
            a := ["positional_encoding", "soft_one_hot"]
        ), f"positional_encoder must be one of {a}"

        if dist_encoding in ["positional_encoding", None]:
            self.positional_encoder = PositionalEncoder(
                n_features, length=length_scale
            )

        self.phi = MLP(n_features, n_features, 4 * n_features)
        self.W = MLP(n_features, n_features, 4 * n_features)


    def forward(self, batch):
        src_node = batch.edge_index[0]
        dst_node = batch.edge_index[1]

        positional_encoding = self.positional_encoder(batch.edge_dist)
        gates, cross_product_gates, scale_edge_dir, scale_features = torch.split(
            self.phi(batch.invariant_node_features[src_node])
            * self.W(positional_encoding),
            self.n_features,
            dim=-1,
        )
        gated_features = multiply_first_dim(
            gates, batch.equivariant_node_features[src_node]
        )
        scaled_edge_dir = multiply_first_dim(
            scale_edge_dir, batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        )

        dst_node_edges = batch.edge_dir.unsqueeze(1).repeat(1, self.n_features, 1)
        dst_equivariant_node_features = batch.equivariant_node_features[dst_node]
        cross_produts = torch.cross(
            dst_node_edges, dst_equivariant_node_features, dim=-1
        )

        gated_cross_products = multiply_first_dim(cross_product_gates, cross_produts)

        dv = scaled_edge_dir + gated_features + gated_cross_products
        ds = multiply_first_dim(scale_features, batch.invariant_node_features[src_node])

        dv = scatter(dv, dst_node, dim=0)
        ds = scatter(ds, dst_node, dim=0)

        batch.equivariant_node_features += dv
        batch.invariant_node_features += ds

        return batch


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class Update(torch.nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
        self.u = EquivariantLinear(n_features, n_features)
        self.v = EquivariantLinear(n_features, n_features)
        self.n_features = n_features
        self.mlp = MLP(2 * n_features, n_features, 3 * n_features)


    def forward(self, batch):
        v = batch.equivariant_node_features
        s = batch.invariant_node_features

        vv = self.v(v)
        uv = self.u(v)

        vv_norm = vv.norm(dim=-1)
        vv_squared_norm = vv_norm ** 2

        mlp_in = torch.cat([vv_norm, s], dim=-1)

        gates, scale_squared_norm, add_invariant_features = torch.split(
            self.mlp(mlp_in), self.n_features, dim=-1
        )

        delta_v = multiply_first_dim(uv, gates)
        delta_s = vv_squared_norm * scale_squared_norm + add_invariant_features

        batch.invariant_node_features = batch.invariant_node_features + delta_s
        batch.equivariant_node_features = batch.equivariant_node_features + delta_v

        return batch


class EquivariantLinear(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super().__init__()
        self.linear = torch.nn.Linear(n_features_in, n_features_out, bias=False)

    def forward(self, x):
        return self.linear(x.swapaxes(-1, -2)).swapaxes(-1, -2)


class Readout(torch.nn.Module):
    def __init__(self, n_features=128, n_features_out=13, n_atoms=20):
        super().__init__()
        self.mlp = MLP(n_features, n_features, 2 * n_features_out)
        self.V = EquivariantLinear(  # pylint:disable=invalid-name
            n_features, n_features_out
        )
        self.n_features_out = n_features_out


    def forward(self, batch):
        invariant_node_features_out, gates = torch.split(
            self.mlp(batch.invariant_node_features), self.n_features_out, dim=-1
        )

        equivariant_node_features = self.V(batch.equivariant_node_features)
        equivariant_node_features_out = multiply_first_dim(
            equivariant_node_features, gates
        )

        batch.invariant_node_features = invariant_node_features_out
        batch.equivariant_node_features = equivariant_node_features_out
        return batch


class PaiNNScore(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
            self,
            batch_size: int = 128,
            n_features=64,
            n_node_types: int = 20,
            n_bond_types: int = 210,
            score_layers=5,
            diff_steps=1000,
            dist_encoding="positional_encoding",
            writhe_layer: bool = True,
            cross_product: bool = False,
            n_atoms: int = 20,
    ):
        super().__init__()
        layers = [
            AddSpatialEdgeFeatures(),
            AddWritheEdges(),
            NominalEmbedding("bonds", n_features, n_types=n_bond_types, feature_type="edge"),
            NominalEmbedding("atoms", n_features, n_types=n_node_types),
            PositionalEmbedding("ts_diff", n_features, diff_steps),
            AddEquivariantFeatures(n_features),
            CombineInvariantFeatures(2 * n_features, n_features),
            PaiNNBase(
                batch_size=batch_size,
                n_features=n_features,
                n_features_out=1,
                n_layers=score_layers,
                dist_encoding=dist_encoding,
                writhe_layer=writhe_layer,
                cross_product=cross_product,
                n_atoms=n_atoms,
            ),
        ]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, batch):
        corr = batch["corr"].clone().to(self.device)
        batch_idx = batch["corr"].batch
        corr.ts_diff = batch["ts_diff"][batch_idx].squeeze()

        dx = self.net(corr).equivariant_node_features.squeeze()
        corr.x += dx

        return corr


# endregion

# region Train

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(data_path: str,
         log_path: str,
         log_name: str,
         ckpt_file: str,
         writhe_layer: bool,
         cross_product: bool,
         n_atoms: int,
         n_features: int,
         n_bond_types: int,
         n_node_types: int,
         n_score_layers: int,
         diffusion_steps: int,
         noise_schedule: str,
         alpha_bar_weight: bool,
         max_epochs: int,
         batch_size: int,
         num_workers: int,
         learning_rate: float,
         accelerator: str,
         n_devices: int,
         strategy: str,
         checkpoint_train_time_interval: int,
         log_every_n_steps: int,
         progress_bar: bool,
         stride: int
         ):
    try:
        del os.environ["SLURM_NTASKS"]
        del os.environ["SLURM_JOB_NAME"]
    except:
        pass
    # make input batch and add features including new writhe feature

    num_devices_available = torch.cuda.device_count()
    if n_devices is None:
        n_devices = num_devices_available
        if n_devices > 1:
            strategy = "ddp"
    else:
        if n_devices > num_devices_available:
            print(f"There are only {num_devices_available} devices but you have requested {n_devices}")
            print(f"Defaulting to {num_devices_available} devices")
            n_devices = num_devices_available
        elif n_devices < num_devices_available:
            print(f"You're using {n_devices} devices but there are {num_devices_available} available")
            print(f"Proceeding with {n_devices} device (won't increase devices automatically")
        else:
            pass

    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)
    if strategy == "dp":
        n_devices = min(8, n_devices)
    if n_devices == 1:
        strategy = None

    dataset = GraphDataSet(file=data_path)[::stride]
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        drop_last=True,
                        )

    logger = Logger(save_dir=log_path,
                    name=log_name)

    checkpoint_callback = ModelCheckpoint(dirpath=f"{log_path}/{log_name}",
                                          filename="{epoch}" + f"_version_{logger.version}",
                                          save_top_k=1,
                                          save_last=True,
                                          train_time_interval=timedelta(minutes=checkpoint_train_time_interval),
                                          )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
        enable_progress_bar=progress_bar,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        logger=logger,
        strategy=strategy,
        devices=n_devices,
        max_steps=-1)

    model = GeometricDDPM(batch_size=batch_size,
                          diffusion_steps=diffusion_steps,
                          noise_schedule=noise_schedule,
                          alpha_bar_weight=alpha_bar_weight,
                          dont_evaluate=True,
                          lr=learning_rate,
                          score_layers=n_score_layers,
                          n_features=n_features,
                          n_bond_types=n_bond_types,
                          n_node_types=n_node_types,
                          writhe_layer=writhe_layer,
                          cross_product=cross_product,
                          n_atoms=n_atoms
                          )

    trainer.fit(model, loader, ckpt_path=ckpt_file)


if __name__ == '__main__':
    # get all of the arguments from the command line
    parser = argparse.ArgumentParser(description="Run DDPM using (chiro)(writhe) PAINN")

    parser.add_argument("--data_path", "-data", required=True, type=str,
                        help="input data for neural network")

    parser.add_argument("--log_path", required=True, type=str,
                        help="path to saved check point file for resuming training, defaults to None and is not required"
                        )

    parser.add_argument("--ckpt_file", default=None, type=str,
                        help="file to a checkpoint to resume training a previous model")

    parser.add_argument("--n_devices", type=int, default=None,
                        help="Number of (cuda enabled) devices to use")

    parser.add_argument("--max_epochs", "-n", type=int, default=100000,
                        help="Number of trials to be used in parameter optimization")

    parser.add_argument("--writhe_layer", "-writhe", type=str2bool, nargs="?",
                        const=True, default=False,
                        help="Use the writhe attention layer in the message passing scheme")

    parser.add_argument("--cross_product", "-cross", type=str2bool, nargs="?",
                        const=True, default=False,
                        help="Use cross product scheme in message layer")

    parser.add_argument("--batch_size", "-batch", default=128, type=int,
                        help="training batch size")

    parser.add_argument("--learning_rate", "-lr", default=1e-3,
                        type=float, help="Learning rate of optimizer")

    parser.add_argument("--accelerator", type=str,
                        default="gpu", help="Type of hardware device to use")

    parser.add_argument("--strategy", type=str, default="ddp",
                        help=("The distributed training method to implement"
                              "options: ddp (Distributed Data Parallel Data), dp (Data Parallel), None (single gpu")
                        )

    parser.add_argument("--n_node_types", type=int, default=20)

    parser.add_argument("--n_bond_types", type=int, default=210)

    parser.add_argument("--diffusion_steps", type=int, default=1000)

    parser.add_argument("--n_features", type=int, default=64)

    parser.add_argument("--n_atoms", type=int, default=20)

    parser.add_argument("--n_score_layers", type=int, default=5)

    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--log_every_n_steps", type=int, default=100)

    parser.add_argument("--alpha_bar_weight", type=str2bool, default=False,
                        const=True, nargs="?")

    parser.add_argument("--noise_schedule", type=str, default="polynomial")

    parser.add_argument("--progress_bar", action='store_false', default=True)

    parser.add_argument("--checkpoint_train_time_interval", "-ckpt_time",
                        type=float, default=20,
                        )

    parser.add_argument("--stride", type=int, default=1,
                        )

    args = parser.parse_args()

    # make a file name to save the best model
    log_name = (
        f"max_epochs:{args.max_epochs}.batch_size:{args.batch_size}.writhe_layer:{str(args.writhe_layer)}."
        f"cross_product:{str(args.cross_product)}.learning_rate:{args.learning_rate}"
    )

    # print(args.ckpt_file)

    # run the main function to train the model
    main(data_path=args.data_path,
         log_path=args.log_path,
         log_name=log_name,
         ckpt_file=args.ckpt_file,
         writhe_layer=args.writhe_layer,
         cross_product=args.cross_product,
         n_atoms=args.n_atoms,
         n_features=args.n_features,
         n_bond_types=args.n_bond_types,
         n_node_types=args.n_node_types,
         n_score_layers=args.n_score_layers,
         diffusion_steps=args.diffusion_steps,
         noise_schedule=args.noise_schedule,
         alpha_bar_weight=args.alpha_bar_weight,
         max_epochs=args.max_epochs,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         learning_rate=args.learning_rate,
         accelerator=args.accelerator,
         n_devices=args.n_devices,
         strategy=args.strategy,
         checkpoint_train_time_interval=args.checkpoint_train_time_interval,
         log_every_n_steps=args.log_every_n_steps,
         progress_bar=args.progress_bar,
         stride=args.stride)

# endregion
