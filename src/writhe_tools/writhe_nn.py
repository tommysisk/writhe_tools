import torch
import torch.nn as nn
from .utils.indexing import get_segments, incidence_writhe_edges


# from torch_scatter import scatter


@torch.jit.script
def divnorm(x: torch.Tensor):
    """Convenience function for (batched) normalization of vectors stored in arrays with last dimension 3"""
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True))


@torch.jit.script
def writhe_segments(xyz: torch.Tensor,
                    segments: torch.Tensor,
                    use_cross: bool = True):
    """
    Compute the writhe (signed crossing) of 2 segment pairs for NSegments and all frames (index 0) in xyz
     (xyz can contain just one frame).

     This computation uses cross products, which are the most precise for small angles.

    Args:
        segments:  tensor of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
                 tensor(Nframes, [seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point]).
                 We have the flexability for segments to simply be one dimensional if only one value of writhe is to be computed.

        xyz: array of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons

        use_cross: compute the writhe from cross products (more accurate, slightly slower)
        or dot products (minor floating point errors for very small angles and single precision but faster, errors )

    #note : this function overwrites transient variables to minimize memory usage,
            this can make the math less interpretable.
    """

    # ensure correct shape for segment for lazy arguments
    assert segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
    segments, xyz = segments.unsqueeze(0) if segments.ndim < 2 else segments, \
        xyz.unsqueeze(0) if xyz.ndim < 3 else xyz

    dx = divnorm((-xyz[:, segments[:, :2], None]
                  + xyz[:, segments[:, None, 2:]]
                  ).reshape(-1, len(segments), 4, 3))

    signs = (torch.cross(xyz[:, segments[:, 3]] - xyz[:, segments[:, 2]],
                         xyz[:, segments[:, 1]] - xyz[:, segments[:, 0]],
                         dim=-1) * dx[:, :, 0]).sum(-1).sign()
    if use_cross:
        dx = divnorm(torch.cross(dx[:, :, [0, 1, 3, 2]],
                                 dx[:, :, [1, 3, 2, 0]],
                                 dim=-1))

        dx = (dx[:, :, [0, 1, 2, 3]] * dx[:, :, [1, 2, 3, 0]]).sum(-1).clamp(-1, 1).arcsin().sum(2)
    else:
        # compute all dot products, then work with scalars
        dx = (dx[:, :, [0, 0, 0, 1, 1, 2]] * dx[:, :, [1, 2, 3, 2, 3, 3]]).sum(-1)
        # get indices; dots is ordered according to indices of 3,3 upper right triangle
        # we compute dots first because we use each twice
        u, v, h = [0, 4, 5, 1], \
                  [4, 5, 1, 0], \
                  [2, 3, 2, 3]
        # surface area from scalars
        dx = ((dx[:, :, u] * dx[:, :, v] - dx[:, :, h])

              / ((1 - dx[:, :, u] ** 2) * (1 - dx[:, :, v] ** 2)).clamp(min=1e-16).sqrt()

              ).clamp(-1, 1).arcsin().sum(-1)

    return dx * signs / (2 * torch.pi)


class TorchWrithe(nn.Module):
    def __init__(self,
                 n_atoms: int = 20,
                 n_features: int = 64,
                 incidence_edges: bool = True):

        super().__init__()
        self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
        self.register_buffer("n_features_", torch.LongTensor([n_features]))
        self.register_buffer("segments", get_segments(n_atoms, tensor=True))
        self.incidence_edges = incidence_edges

    @property
    def incidence_edges(self):
        return self.incidence_edges_

    @incidence_edges.setter
    def incidence_edges(self, x):
        self.incidence_edges_ = x
        if x and not hasattr(self, "indices"):
            n_atoms = self.n_atoms

            self.register_buffer("indices", torch.unique(incidence_writhe_edges(n_atoms),
                                                         dim=1,
                                                         return_inverse=True
                                                         )[-1]
                                 )

            self.register_buffer("zeros", torch.zeros(int((n_atoms ** 2 - n_atoms) / 2 - 2)))

            self.register_buffer("sort", torch.einsum("ij,i->j",
                                                      torch.cat([torch.triu_indices(n_atoms, n_atoms, 1),
                                                                 torch.triu_indices(n_atoms, n_atoms, 1).flip(0)],
                                                                dim=-1).float(),
                                                      torch.Tensor([n_atoms, 1]).float()
                                                      ).argsort().squeeze())
            # self.register_buffer("repeat_interleave",
            #                      torch.arange(int(((n_atoms - 1) * (n_atoms - 2) / 2 ) - (n_atoms - 2))).repeat_interleave(4)
            #                      )

    @property
    def n_atoms(self):
        return self.n_atoms_.item()

    @property
    def n_features(self):
        return self.n_features_.item()

    def laplacian_(self, wr):
        assert wr.ndim == 2, "arg 'wr' must be 2D"  # Case: No extra dimension (scalar sum)
        triu = self.zeros.repeat(len(wr), 1).scatter_add_(1,
                                                          self.indices.repeat(len(wr), 1),
                                                          wr.repeat_interleave(4, 1))

        return torch.nn.functional.pad(triu, (1, 1), 'constant', 0.0).repeat(1, 2)[:, self.sort]

    def laplacian(self, wr):
        if wr.ndim == 2:  # Case: No extra dimension (scalar sum)
            return self.laplacian_(wr)
        else:  # Case: Extra dimension present (vector sum)
            return torch.stack([self.laplacian_(i) for i in wr.permute(2, 0, 1)], dim=-1)

    def compute_writhe_vector(self, xyz):
        assert self.segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
        segments, xyz = self.segments.unsqueeze(0) if self.segments.ndim < 2 else self.segments, \
            xyz.unsqueeze(0) if xyz.ndim < 3 else xyz

        dx = divnorm((-xyz[:, segments[:, :2], None]
                      + xyz[:, segments[:, None, 2:]]
                      ).reshape(-1, len(segments), 4, 3))

        axial_vector = torch.cross(xyz[:, segments[:, 3]] - xyz[:, segments[:, 2]],
                                   xyz[:, segments[:, 1]] - xyz[:, segments[:, 0]],
                                   dim=-1)


        signs = (axial_vector * dx[:, :, 0]).sum(-1).sign()

        dx = divnorm(torch.cross(dx[:, :, [0, 1, 3, 2]],
                                 dx[:, :, [1, 3, 2, 0]],
                                 dim=-1))

        dx = (dx[:, :, [0, 1, 2, 3]] * dx[:, :, [1, 2, 3, 0]]).sum(-1).clamp(-1, 1).arcsin().sum(2)

        return dx * signs / (2 * torch.pi), axial_vector

    def forward(self, xyz, vector: bool = False):

        if vector:
            return [self.laplacian(i) if self.incidence_edges else i for i in
                    self.compute_writhe_vector(xyz=xyz.reshape(-1, self.n_atoms, 3))]
        else:
            writhe = writhe_segments(xyz=xyz.reshape(-1, self.n_atoms, 3), segments=self.segments)
            return self.laplacian(writhe) if self.incidence_edges else writhe


class AddWritheEdges(nn.Module):
    def __init__(self, n_atoms=20, n_features=64, add_vector: bool = False):
        super().__init__()
        self.writhe = TorchWrithe(n_atoms=n_atoms,
                                  n_features=n_features)
        self.add_vector = add_vector

    def forward(self, batch):
        if self.add_vector:
            # noinspection PyCallingNonCallable
            batch.writhe, batch.vector = self.writhe(batch.x, vector=self.add_vector)
            # batch.writhe, batch.vector = wr, vector
            return batch
        else:
            batch.writhe = self.writhe(batch.x)
            return batch




# class TorchWrithe(nn.Module):
#     def __init__(self,
#                  n_atoms: int = 20,
#                  n_features: int = 64,
#                  incidence_edges: bool = True):
#
#         super().__init__()
#         self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
#         self.register_buffer("n_features_", torch.LongTensor([n_features]))
#         self.register_buffer("segments", get_segments(n_atoms, tensor=True))
#
#         if incidence_edges:
#             self.incidence_edges = True
#             self.register_buffer("indices",
#                                  torch.unique(incidence_laplacian(n_atoms),
#                                               dim=1,
#                                               return_inverse=True)[-1].unsqueeze(-1)
#                                  )
#
#            # self.register_buffer("triu", torch.triu_indices(self.n_atoms, self.n_atoms, 1))
#            # unsort_full_edges = torch.cat([self.triu, self.triu.flip(0)], dim=-1)
#
#             self.register_buffer("zeros", torch.zeros(int((n_atoms ** 2 - n_atoms) / 2 - 2)))
#             # we need to comply with row-wise ordering (PaiNN) after computing the triu of the Laplacian
#             self.register_buffer("sort",
#                                  torch.einsum("ij,i->j",
#                                               torch.cat([torch.triu_indices(self.n_atoms, self.n_atoms, 1),
#                                                         torch.triu_indices(self.n_atoms, self.n_atoms, 1).flip(0)],
#                                                         dim=-1).float(),
#                                              # unsort_full_edges.float(),
#                                               torch.Tensor([n_atoms, 1]).float()
#                                               ).argsort().squeeze())
#         else:
#             self.incidence_edges = False
#
#     @property
#     def n_atoms(self):
#         return self.n_atoms_.item()
#
#     @property
#     def n_features(self):
#         return self.n_features_.item()
#
#     def forward(self, x):
#         wr = writhe_segments(xyz=x.reshape(-1, self.n_atoms, 3),
#                             segments=self.segments)
#
#         if self.incidence_edges:
#             # compute the writhe graph Laplacian : B @ W @ B.T
#             # we only do this with non-redundant, flattened versions of the actual writhe matrix
#             # the computation is expanded to the required dimensions after
#             triu = torch.nn.functional.pad(
#                 self.zeros.repeat(len(wr), 1).T.scatter_add_(0,
#                                                              self.indices.expand(-1, len(wr)),
#                                                              wr.T.repeat_interleave(4, dim=0)).T,
#                 (1, 1),
#                 mode='constant',
#                 value=0.).T
#
#             return torch.cat([triu, triu])[self.sort].T.flatten()
#
#         else:
#             return wr


# class WritheMessage(nn.Module):
#     """
#     Expedited Writhe message layer.
#
#     """
#
#     def __init__(self,
#                  n_atoms: int,
#                  n_features: int,
#                  batch_size: int,
#                  bins: int = 300,
#                  node_attr: str = "invariant_node_features",
#                  node_indices: torch.Tensor = None,
#                  edge_attr: str = "invariant_edge_features",
#                  distance_attr: str = "edge_dist",
#                  distance_attention: bool = True,
#                  sin_embedding: bool = True,
#                  incidence_edges: bool = True,
#                  # dir_attr: str = "edge_dir",
#                  segment_length: int = 1,
#                  gaussian_bins: bool = False,
#                  bin_low: float = -1,
#                  bin_high: float = 1,
#                  bin_std: float = 1,
#                  bin_mean: float = 0,
#                  residual: bool = True
#                  ):
#
#         super().__init__()
#
#         # from torch_scatter import scatter
#         # prerequisit information
#         segments = get_segments(n_atoms, index0=node_indices, length=segment_length, tensor=True)
#         # need edges based on the segments the writhe is computed from
#         edges = torch.cat([segments[:, [0, 2]], segments[:, [1, 3]]]).T if not incidence_edges\
#                 else incidence_laplacian(n_atoms)
#         # edges = torch.cat([edges, torch.flip(edges, (0,))], dim=1) # this doesn't help
#
#         self.register_buffer("edge_indices",
#                              (flat_index(*edges, n=n_atoms, d0=1).repeat(batch_size, 1)
#                               + int(n_atoms ** 2 - n_atoms) * torch.arange(batch_size).unsqueeze(-1)
#                               ).flatten()
#                              )
#
#         self.register_buffer("edges",
#                              torch.cat([i * n_atoms + edges for i in range(batch_size)], axis=1).long())
#
#         self.register_buffer("segments", segments)
#         self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
#         self.register_buffer("batch_size_", torch.LongTensor([batch_size]))
#         self.register_buffer("segment_length", torch.LongTensor([segment_length]))
#         self.register_buffer("distance_attention_", torch.LongTensor([int(distance_attention)]))
#         self.register_buffer("sin_embedding_", torch.LongTensor([int(sin_embedding)]))
#         self.register_buffer("incidence_edges_", torch.LongTensor([int(incidence_edges)]))
#
#         self.node_attr = node_attr
#         self.edge_attr = edge_attr
#         self.distance_attr = distance_attr
#         # self.dir_attr = dir_attr
#         self.n_features = n_features
#         self.residual = residual
#         self.bin_low = bin_low
#         self.bin_high = bin_high
#
#         # writhe embedding
#         if not sin_embedding:
#             # learned embedding of writhe values based on soft-one-hot as co-effs in combo of learned basis functions
#             self.soft_one_hot = GaussEncoder(low=self.bin_low,
#                                              high=self.bin_high,
#                                              number=bins,
#                                              gauss=gaussian_bins,
#                                              std=bin_std,
#                                              mu=bin_mean)
#
#             std = 1. / math.sqrt(n_features)  # 2 / bins,  this is how torch initializes embedding layers
#
#             self.register_parameter("basis",
#                                     torch.nn.Parameter(
#                                         torch.Tensor(1, 1, bins, n_features).uniform_(-std, std),  # normal_(0, std)
#                                         requires_grad=True)
#                                     )
#         if not distance_attention:
#             # learned GAT attention
#             # will use writhe embedding as edge feature if  edge_attr is not present in graph inputs
#             self.attention = nn.Sequential(nn.Linear(n_features * 3, n_features, bias=False),
#                                            nn.SiLU(),
#                                            nn.Linear(n_features, 1, bias=False)
#                                            )
#
#     @property
#     def n_atoms(self):
#         return self.n_atoms_.item()
#
#     @property
#     def distance_attention(self):
#         return self.distance_attention_.item()
#
#     @property
#     def sin_embedding(self):
#         return self.sin_embedding_.item()
#
#     @property
#     def incidence_edges(self):
#         return self.incidence_edges_.item()
#
#     def embed_writhe(self, wr):
#         if self.sin_embedding:
#             embed = torch.stack([torch.sin(wr / 10 * i * torch.pi) for i in torch.arange(self.n_features)],
#                                dim=-1)
#
#         else:
#             embed = torch.sum(self.soft_one_hot(wr).unsqueeze(-1) * self.basis, dim=-2)
#
#         return (embed.repeat_interleave(4, dim=1) if self.incidence_edges else
#                embed.repeat(1, 2, 1)).reshape(-1, self.n_features)
#
#     def compute_writhe(self, x):
#
#         return self.embed_writhe(
#             writhe_segments(xyz=x.x.reshape(-1, self.n_atoms, 3),
#                             segments=self.segments))
#
#     def forward(self, x, update=True):
#
#         src_node, dst_node = (i.flatten() for i in self.edges)
#         writhe = self.compute_writhe(x)
#
#         if self.distance_attention:
#             # distances attention will use -d_{i, j}**2 as the attention logits
#
#             weights = (getattr(x, self.distance_attr) if hasattr(x, str(self.distance_attr))
#                        else (x.x[src_node] - x.x[dst_node]).norm(dim=-1)
#                        )[self.edge_indices].pow(2).neg().exp()
#         else:
#             weights = self.attention(
#                 torch.cat([
#                     x[self.node_attr][src_node],
#                     x[self.node_attr][dst_node],
#                     x[self.edge_attr][self.edge_indices] \
#                         if hasattr(x, str(self.edge_attr)) else writhe
#                 ], 1)).flatten().exp()
#
#         # get attention weights, softmax normalize, scatter
#         attention = (weights / scatter(weights, dst_node)[dst_node]).unsqueeze(-1)
#         message = scatter(writhe * attention, dst_node, dim=0)
#
#         if update:
#             x[self.node_attr] = x[self.node_attr] + message if self.residual else message
#             return x
#         else:
#             return x[self.node_attr] + message if self.residual else message
#
#
# class GaussEncoder(nn.Module):
#     def __init__(self,
#                  low: float,
#                  high: float,
#                  number: int,
#                  gauss: bool = False,
#                  std: float = 1,
#                  mu: float = 0):
#
#         super().__init__()
#
#         if gauss:
#             self.register_buffer("step_", torch.diff(gaussian_binning(low, high, number, std, mu)))
#             self.register_buffer("bin_centers", gaussian_binning(low, high, number - 1, std, mu))
#
#         else:
#             bins = torch.linspace(low, high, number)
#             self.register_buffer("step_", torch.Tensor([bins[1] - bins[0]]))
#             self.register_buffer("bin_centers", bins)
#
#     @property
#     def step(self):
#         return self.step_.item() if len(self.step_) == 1 else self.step_
#
#     def forward(self, x):
#
#         diff = (x[..., None] - self.bin_centers) / self.step
#
#         return diff.pow(2).neg().exp().div(1.12)
#
#
# def gaussian_binning(low: float,
#                      high: float,
#                      number: int,
#                      std: float = 1,
#                      mean: float = 0):
#     """
#     Creates symmetric non-uniform bins where bin
#     widths are proportional to the height of a Gaussian distribution.
#
#     Parameters:
#     - low: low of the interval.
#     - high: high of the interval.
#     - number: Number of bins.
#     - std: Standard deviation of the Gaussian distribution.
#     - mean: Mean of the Gaussian distribution.
#
#     Returns:
#     - bin_edges: Tensor of bin edges (length: num_bins + 1), symmetric and in increasing order.
#     """
#     number += 2
#     # Generate a Gaussian PDF over the specified symmetric range
#     x = torch.linspace(low, high, 500000)  # we only compute this once (yee haw)
#     pdf = torch.distributions.Normal(loc=mean, scale=std).log_prob(x).exp()
#     # Normalize the PDF so that it sums to 1
#     pdf /= torch.sum(pdf)
#     # Create a cumulative sum to represent the integral of the PDF
#     pdf_cumsum = torch.cumsum(pdf, dim=0)
#     # Target cumulative values for bin edges (0 to 1) evenly spaced
#     target_cdf_vals = torch.linspace(0 - 1 / number,
#                                      1 + 1 / number,
#                                      number + 1)
#
#     # Find where the target CDF values map onto the cumulative PDF
#     indices = torch.searchsorted(pdf_cumsum, target_cdf_vals).clamp(0, len(x) - 1)
#
#     # Linear interpolation for bin edge determination
#     x0 = x[indices - 1].clamp(min=low)  # Lower bound for interpolation
#     x1 = x[indices].clamp(max=high)  # Upper bound for interpolation
#     y0 = pdf_cumsum[indices - 1].clamp(min=0)  # Corresponding CDF values
#     y1 = pdf_cumsum[indices].clamp(max=1)  # Next CDF values
#
#     # Linear interpolation formula for bin edges
#     bin_edges = x0 + (target_cdf_vals - y0) * (x1 - x0) / (y1 - y0 + 1e-9)
#
#     # To create symmetric bin edges about the mean
#     left_half = mean - (bin_edges - mean)  # Reflect bin edges about the mean
#     full_bin_edges = torch.cat((left_half.flip(0), bin_edges))
#
#     # Ensure we only take 'number + 1' edges and sort them
#     full_bin_edges = full_bin_edges[:number]
#     final_bin_edges = torch.sort(full_bin_edges)[0]
#
#     return final_bin_edges[1:]

# cross product method to compute the writhe (Klenin 2000)


# class WritheMessage(nn.Module):
#     """
#     Expedited Writhe message layer.
#
#     """
#
#     def __init__(self,
#                  n_atoms: int,
#                  n_features: int,
#                  batch_size: int,
#                  bins: int = 100,
#                  node_feature: str = "invariant_node_features",
#                  segment_length: int = 1,
#                  bin_start: float = 1,
#                  bin_end: float = -1,
#                  activation: callable = nn.LeakyReLU,
#                  residual: bool = True
#                  ):
#
#         super().__init__()
#
#         # prerequisit information
#         segments = get_segments(n_atoms, length=segment_length, tensor=True)
#         edges = torch.cat([segments[:, [0, 2]], segments[:, [1, 3]]]).T
#         # edges = torch.cat([edges, torch.flip(edges, (0,))], dim=1) # this doesn't help
#
#         self.register_buffer("edges", torch.cat([i * n_atoms + edges for i in range(batch_size)], axis=1).long())
#         self.register_buffer("segments", segments)
#         self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
#         self.register_buffer("batch_size_", torch.LongTensor([batch_size]))
#         self.register_buffer("segment_length", torch.LongTensor([segment_length]))
#
#         self.node_feature = node_feature
#         self.n_features = n_features
#         self.residual = residual
#         self.bin_start = bin_start
#         self.bin_end = bin_end
#
#         # writhe embedding
#
#         self.soft_one_hot = partial(soft_one_hot_linspace,
#                                     start=self.bin_start,
#                                     end=self.bin_end,
#                                     number=bins,
#                                     basis="gaussian",
#                                     cutoff=False)
#
#         std = 1. / math.sqrt(n_features) #2 / bins  # # this is how torch initializes embedding layers
#
#         self.register_parameter("basis",
#                                 torch.nn.Parameter(torch.Tensor(1, 1, bins, n_features).uniform_(-std, std), #normal_(0, std)
#                                                    requires_grad=True)
#                                 )
#
#         # attention mechanism
#
#         self.query = nn.Sequential(nn.Linear(n_features, n_features), activation())
#         self.key = nn.Sequential(nn.Linear(n_features, n_features), activation())
#         self.value = nn.Sequential(nn.Linear(n_features, n_features), activation())
#
#         self.attention = nn.Sequential(nn.Linear(int(3 * n_features), 1), activation())
#
#     @property
#     def n_atoms(self):
#         return self.n_atoms_.item()
#
#     def embed_writhe(self, wr):
#         return (self.soft_one_hot(wr).unsqueeze(-1) * self.basis).sum(-2)
#
#     def compute_writhe(self, x):
#         return self.embed_writhe(writhe_segments(xyz=x.x.reshape(-1, self.n_atoms, 3),
#                                                  segments=self.segments,)
#                                  ).repeat(1, 2, 1).reshape(-1, self.n_features)
#
#     def forward(self, x, update=True):
#         features = getattr(x, self.node_feature).clone()
#
#         src_node, dst_node = (i.flatten() for i in self.edges)
#
#         writhe = self.compute_writhe(x)
#
#         attention_input = torch.cat([getattr(self, i)(j) for i, j in
#                                      zip(["query", "key", "value"], [features[dst_node], features[src_node], writhe])
#                                      ], dim=-1)
#
#         logits = self.attention(attention_input).flatten()
#
#         logits = logits - logits.max()  # added numerical stability without effecting output by properties of softmax
#
#         weights = torch.exp(logits)
#
#         attention = (weights / scatter(weights, dst_node)[dst_node]).unsqueeze(-1)
#
#         message = scatter(writhe * attention, dst_node, dim=0)
#
#         if update:
#
#             x[self.node_feature] = features + message if self.residual else message
#
#             return x
#
#         else:
#             return features + message if self.residual else message


# class _SoftUnitStep(torch.autograd.Function):
#     # pylint: disable=arguments-differ
#
#     @staticmethod
#     def forward(ctx, x) -> torch.Tensor:
#         ctx.save_for_backward(x)
#         y = torch.zeros_like(x)
#         m = x > 0.0
#         y[m] = (-1 / x[m]).exp()
#         return y
#
#     @staticmethod
#     def backward(ctx, dy) -> torch.Tensor:
#         (x,) = ctx.saved_tensors
#         dx = torch.zeros_like(x)
#         m = x > 0.0
#         xm = x[m]
#         dx[m] = (-1 / xm).exp() / xm.pow(2)
#         return dx * dy
#
#
# def soft_unit_step(x):
#     r"""smooth :math:`C^\infty` version of the unit step function
#
#     .. math::
#
#         x \mapsto \theta(x) e^{-1/x}
#
#
#     Parameters
#     ----------
#     x : `torch.Tensor`
#         tensor of shape :math:`(...)`
#
#     Returns
#     -------
#     `torch.Tensor`
#         tensor of shape :math:`(...)`
#
#     """
#     return _SoftUnitStep.apply(x)
#
#
# def soft_one_hot_linspace(x: torch.Tensor, start, end, number, basis=None, cutoff=None) -> torch.Tensor:
#     r"""Projection on a basis of functions
#
#     Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,
#
#     .. math::
#
#         y_i(x) = \frac{1}{Z} f_i(x)
#
#     where :math:`x` is the input and :math:`f_i` is the ith basis function.
#     :math:`Z` is a constant defined (if possible) such that,
#
#     .. math::
#
#         \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1
#
#     See the last plot below.
#     Note that ``bessel`` basis cannot be normalized.
#
#     Parameters
#     ----------
#     x : `torch.Tensor`
#         tensor of shape :math:`(...)`
#
#     start : float
#         minimum value span by the basis
#
#     end : float
#         maximum value span by the basis
#
#     number : int
#         number of basis functions :math:`N`
#
#     basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
#         choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of
#         other basis choices
#
#     cutoff : bool
#         if ``cutoff=True`` then for all :math:`x` outside of the interval defined by ``(start, end)``,
#         :math:`\forall i, \; f_i(x) \approx 0`
#
#     Returns
#     -------
#     `torch.Tensor`
#         tensor of shape :math:`(..., N)`
#
#     """
#     # pylint: disable=misplaced-comparison-constant
#
#     if cutoff not in [True, False]:
#         raise ValueError("cutoff must be specified")
#
#     if not cutoff:
#         values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
#         step = values[1] - values[0]
#     else:
#         values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
#         step = values[1] - values[0]
#         values = values[1:-1]
#
#     diff = (x[..., None] - values) / step
#
#     if basis == "gaussian":
#         return diff.pow(2).neg().exp().div(1.12)
#
#     if basis == "cosine":
#         return torch.cos(math.pi / 2 * diff) * (diff < 1) * (-1 < diff)
#
#     if basis == "smooth_finite":
#         return 1.14136 * torch.exp(torch.tensor(2.0)) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)
#
#     if basis == "fourier":
#         x = (x[..., None] - start) / (end - start)
#         if not cutoff:
#             i = torch.arange(0, number, dtype=x.dtype, device=x.device)
#             return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
#         else:
#             i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
#             return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)
#
#     if basis == "bessel":
#         x = x[..., None] - start
#         c = end - start
#         bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
#         out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x
#
#         if not cutoff:
#             return out
#         else:
#             return out * ((x / c) < 1) * (0 < x)
#
#     raise ValueError(f'basis="{basis}" is not a valid entry')
#
#
#
# @torch.jit.script
# def writhe_segments_dot(xyz: torch.Tensor,
#                         segments: torch.Tensor,
#                         double: bool = False):
#     """
#     Compute the writhe (signed crossing) of 2 segment pairs for NSegments and all frames (index 0) in xyz
#     (xyz can contain just one frame).
#
#     This computation trades speed for precision. In cases where very small angles are encountered,
#     this approach has more error than the methods using cross products. However,
#     it's worth noting that this formulation is analytically correct and that the issues with precision
#     stem from computing magnitudes of cross products and determinants with the gram determinant
#     rather than scalar triple products.
#
#     When using double precision, this approach gives results that are agree with the cross product method up to ~ 1e-5
#
#     **Arguments**:
#     - segments: Array of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:
#       array(Nframes, [seg1_start_point, seg1_end_point, seg2_start_point, seg2_end_point]).
#       Segments can be one-dimensional if only one value of writhe is to be computed.
#     - xyz: Array of shape (Nframes, N_alpha_carbons, 3), coordinate array giving the positions of all alpha carbons.
#
#     The program uses dot products and trigonometric identities to refactor the computation, making it
#     ~twice as fast as the cross product formulation.
#
#     It makes use of the following identities :
#
#     Assume that a, b, c, d are normalized vectors in R^3:
#
#          a.cross(b) · c.cross(d)  /  ||a.cross(b)|| * ||c.cross(d)||
#
#
#         = [(a · c) * (b · d) - (a · d) * (b · c)] / sqrt( (1 - (a · b)**2) * (1 - (c · d)**2) )
#
#     In the special case where b == c:
#
#         = [(a · b) * (b · d) - (a · d) * ||b||**2 ] / sqrt( (1 - (a · b)**2) * (1 - (b · d)**2) )
#
#     Note that ||·||**2 == 1.
#
#     Set
#     u = (a · b),
#     v = (b · d),
#     h = (a · d)
#     and define the scalar function:
#
#        sin(theta - pi / 2) = (u * v - h) / sqrt( (1 - u**2) * (1 - v**2) )
#
#     Precomputing dot products and recursive application of this formula to compute the
#     interior angles of the spherical quadrilateral allows computation of the writhe
#     with 6 dot products and a determinant per crossing.
#     """
#     # ensure shape for segments
#     assert segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
#     dtype = xyz.dtype
#     # catch any lazy arguments
#     segments, xyz = segments.unsqueeze(0) if segments.ndim < 2 else segments, \
#         xyz.unsqueeze(0) if xyz.ndim < 3 else xyz
#     xyz = xyz.double() if double else xyz
#
#     # displacement vectors between end points
#     dx = divnorm((-xyz[:, segments[:, :2], None] + xyz[:, segments[:, None, 2:]]
#                   ).reshape(-1, len(segments), 4, 3))
#
#     # compute all dot products, then work with scalars
#     dots = (dx[:, :, [0, 0, 0, 1, 1, 2]] * dx[:, :, [1, 2, 3, 2, 3, 3]]).sum(-1)
#
#     # get indices; dots is ordered according to indices of 3,3 upper right triangle
#     # NOTE : we compute dots first because we use each twice
#     u, v, h = [0, 4, 5, 1], \
#         [4, 5, 1, 0], \
#         [2, 3, 2, 3]
#
#     # surface area from scalars
#     dots = ((dots[:, :, u] * dots[:, :, v] - dots[:, :, h])
#
#             / ((1 - dots[:, :, u] ** 2) * (1 - dots[:, :, v] ** 2)).clamp(min=1e-15).sqrt()
#
#             ).clip(-1, 1).arcsin().sum(-1)
#
#     # signs from triple products
#     signs = (torch.cross(xyz[:, segments[:, 3]] - xyz[:, segments[:, 2]],
#                          xyz[:, segments[:, 1]] - xyz[:, segments[:, 0]],
#                          dim=-1) * dx[:, :, 0]).sum(-1).sign()
#
#     return torch.squeeze(dots * signs).to(dtype) / (2 * torch.pi)
