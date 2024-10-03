#!/usr/bin/env python
from typing import List, Any
import torch
import torch.nn as nn
from torch_scatter import scatter
import math
from .writhe_utils import get_segments
from functools import partial


@torch.jit.script
def nnorm(x: torch.Tensor):
    """Convenience function for (batched) normalization of vectors stored in arrays with last dimension 3"""
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)


@torch.jit.script
def ncross(x: torch.Tensor, y: torch.Tensor):
    """Convenience function for (batched) cross products of vectors stored in arrays with last dimension 3"""
    return torch.cross(x, y, dim=-1)


@torch.jit.script
def ndot(x: torch.Tensor, y: torch.Tensor):
    """Convenience function for (batched) dot products of vectors stored in arrays with last dimension 3"""
    return torch.sum(x * y, dim=-1)


@torch.jit.script
def ndet(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor):
    """for the triple product and finding the signed sin of the angle between v2 and v3, v1 should
    be set equal to a vector mutually orthogonal to v2,v3"""
    return ndot(v1, ncross(v2, v3))


@torch.jit.script
def uproj(a, b, norm_b: bool = True, norm_proj: bool = True):
    b = nnorm(b) if norm_b else b
    # faster than np.matmul when using ray
    proj = a - b * (a * b).sum(dim=-1).unsqueeze(-1)
    return nnorm(proj) if norm_proj else proj


@torch.jit.script
def writhe_segments(xyz: torch.Tensor, segments: torch.Tensor,):

    """
    compute the writhe (signed crossing) of 2 segment pairs for NSegments and all frames (index 0) in xyz (xyz can contain just one frame)

    **provide both of the following**

    segments:  array of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
             array(Nframes, [seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point]).
             We have the flexability for segments to simply be one dimensional if only one value of writhe is to be computed.

    xyz: array of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons

    """
    # ensure correct shape for segment for lazy arguments
    # if smat is None:
    assert segments is not None and xyz is not None, "Must provide segments and xyz if not smat"
    assert segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
    segments = segments.unsqueeze(0) if segments.ndim < 2 else segments
    smat = (xyz.unsqueeze(0) if xyz.ndim < 3 else xyz)[:, segments]

    displacements = nnorm((-smat[:, :, :2, None, :] + smat[:, :, None, 2:, :]
                           ).reshape(-1, smat.shape[1], 4, 3))

    crosses = nnorm(ncross(displacements[:, :, [0, 1, 3, 2]], displacements[:, :, [1, 3, 2, 0]]))

    omega = torch.arcsin(ndot(crosses[:, :, [0, 1, 2, 3]],
                              crosses[:, :, [1, 2, 3, 0]]).clip(-1, 1)).sum(2)

    signs = torch.sign(ndot(ncross(smat[:, :, 3] - smat[:, :, 2],
                                   smat[:, :, 1] - smat[:, :, 0]),
                            displacements[:, :, 0]))

    return torch.squeeze(omega * signs) / (2 * torch.pi)


class WritheMessage(nn.Module):
    """
    Expedited Writhe message layer.

    """

    def __init__(self,
                 n_atoms: int,
                 n_features: int,
                 batch_size: int,
                 bins: int = 20,
                 node_feature: str = "invariant_node_features",
                 segment_length: int = 1,
                 bin_start: float = 1,
                 bin_end: float = -1,
                 activation: callable = nn.LeakyReLU,
                 residual: bool = True
                 ):

        super().__init__()

        # prerequisit information
        segments = get_segments(n_atoms, length=segment_length, tensor=True)
        edges = torch.cat([segments[:, [0, 2]], segments[:, [1, 3]]]).T
        # edges = torch.cat([edges, torch.flip(edges, (0,))], dim=1) # this doesn't help

        self.register_buffer("edges", torch.cat([i * n_atoms + edges for i in range(batch_size)], axis=1).long())
        self.register_buffer("segments", segments)
        self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
        self.register_buffer("batch_size_", torch.LongTensor([batch_size]))
        self.register_buffer("segment_length", torch.LongTensor([segment_length]))

        self.node_feature = node_feature
        self.n_features = n_features
        self.residual = residual
        self.bin_start = bin_start
        self.bin_end = bin_end

        # writhe embedding

        self.soft_one_hot = partial(soft_one_hot_linspace,
                                    start=self.bin_start,
                                    end=self.bin_end,
                                    number=bins,
                                    basis="gaussian",
                                    cutoff=False)

        std = 2 / bins  # 1. / math.sqrt(n_features) # this is how torch initializes embedding layers

        self.register_parameter("basis",
                                torch.nn.Parameter(torch.Tensor(1, 1, bins, n_features).normal_(0, std),
                                                   # uniform_(-std, std),
                                                   requires_grad=True)
                                )

        # attention mechanism

        self.query = nn.Sequential(nn.Linear(n_features, n_features), activation())
        self.key = nn.Sequential(nn.Linear(n_features, n_features), activation())
        self.value = nn.Sequential(nn.Linear(n_features, n_features), activation())

        self.attention = nn.Sequential(nn.Linear(int(3 * n_features), 1), activation())

    @property
    def n_atoms(self):
        return self.n_atoms_.item()

    def embed_writhe(self, wr):
        return (self.soft_one_hot(wr).unsqueeze(-1) * self.basis).sum(-2)

    def compute_writhe(self, x):
        return self.embed_writhe(writhe_segments(xyz=x.x.reshape(-1, self.n_atoms, 3),
                                                 segments=self.segments,)
                                 ).repeat(1, 2, 1).reshape(-1, self.n_features)

    def forward(self, x, update=True):
        features = getattr(x, self.node_feature).clone()

        src_node, dst_node = (i.flatten() for i in self.edges)

        writhe = self.compute_writhe(x)

        attention_input = torch.cat([getattr(self, i)(j) for i, j in
                                     zip(["query", "key", "value"], [features[dst_node], features[src_node], writhe])
                                     ], dim=-1)

        logits = self.attention(attention_input).flatten()

        logits = logits - logits.max()  # added numerical stability without effecting output by properties of softmax

        weights = torch.exp(logits)

        attention = (weights / scatter(weights, dst_node)[dst_node]).unsqueeze(-1)

        message = scatter(writhe * attention, dst_node, dim=0)

        if update:

            x[self.node_feature] = features + message if self.residual else message

            return x

        else:
            return features + message if self.residual else message



class _SoftUnitStep(torch.autograd.Function):
    # pylint: disable=arguments-differ

    @staticmethod
    def forward(ctx, x) -> torch.Tensor:
        ctx.save_for_backward(x)
        y = torch.zeros_like(x)
        m = x > 0.0
        y[m] = (-1 / x[m]).exp()
        return y

    @staticmethod
    def backward(ctx, dy) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        dx = torch.zeros_like(x)
        m = x > 0.0
        xm = x[m]
        dx[m] = (-1 / xm).exp() / xm.pow(2)
        return dx * dy


def soft_unit_step(x):
    r"""smooth :math:`C^\infty` version of the unit step function

    .. math::

        x \mapsto \theta(x) e^{-1/x}


    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(...)`

    """
    return _SoftUnitStep.apply(x)


def soft_one_hot_linspace(x: torch.Tensor, start, end, number, basis=None, cutoff=None) -> torch.Tensor:
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis

    end : float
        maximum value span by the basis

    number : int
        number of basis functions :math:`N`

    basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
        choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of
        other basis choices

    cutoff : bool
        if ``cutoff=True`` then for all :math:`x` outside of the interval defined by ``(start, end)``,
        :math:`\forall i, \; f_i(x) \approx 0`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., N)`

    """
    # pylint: disable=misplaced-comparison-constant

    if cutoff not in [True, False]:
        raise ValueError("cutoff must be specified")

    if not cutoff:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
    else:
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step

    if basis == "gaussian":
        return diff.pow(2).neg().exp().div(1.12)

    if basis == "cosine":
        return torch.cos(math.pi / 2 * diff) * (diff < 1) * (-1 < diff)

    if basis == "smooth_finite":
        return 1.14136 * torch.exp(torch.tensor(2.0)) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)

    if basis == "fourier":
        x = (x[..., None] - start) / (end - start)
        if not cutoff:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        else:
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == "bessel":
        x = x[..., None] - start
        c = end - start
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if not cutoff:
            return out
        else:
            return out * ((x / c) < 1) * (0 < x)

    raise ValueError(f'basis="{basis}" is not a valid entry')

