import torch
import torch.nn as nn
import math
from typing import Optional
from .utils.indexing import get_segments, flat_index


@torch.jit.script
def divnorm(x: torch.Tensor):
    """Convenience function for (batched) normalization of vectors stored in arrays with last dimension 3"""
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)


@torch.jit.script
def writhe_segments_cross(xyz: torch.Tensor, segments: torch.Tensor,):

    """
    Compute the writhe (signed crossing) of 2 segment pairs for NSegments and all frames (index 0) in xyz
     (xyz can contain just one frame).

     This computation uses cross products, which are the most precise for small angles.

    Args:
        segments:  tensor of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
                 tensor(Nframes, [seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point]).
                 We have the flexability for segments to simply be one dimensional if only one value of writhe is to be computed.

        xyz: array of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons

    """

    # ensure correct shape for segment for lazy arguments
    assert segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
    segments, xyz = segments.unsqueeze(0) if segments.ndim < 2 else segments, \
                    xyz.unsqueeze(0) if xyz.ndim < 3 else xyz

    dx = divnorm((-xyz[:, segments[:, :2], None]
                  + xyz[:, segments[:, None, 2:]]
                  ).reshape(-1, len(segments), 4, 3))

    crosses = divnorm(torch.cross(dx[:, :, [0, 1, 3, 2]],
                                  dx[:, :, [1, 3, 2, 0]],
                                  dim=-1))

    # this will be the sum of the angles or surface area element,
    # we reassign the crosses variable to clear memory
    crosses = (crosses[:, :, [0, 1, 2, 3]]
               * crosses[:, :, [1, 2, 3, 0]]
               ).sum(-1).clip(-1, 1).arcsin().sum(2)

    signs = (torch.cross(xyz[:, segments[:, 3]] - xyz[:, segments[:, 2]],
                         xyz[:, segments[:, 1]] - xyz[:, segments[:, 0]],
                         dim=-1) * dx[:, :, 0]).sum(-1).sign()

    return torch.squeeze(crosses * signs) / (2 * torch.pi)


@torch.jit.script
def writhe_segments_cross_index(xyz: torch.Tensor,
                                segments: torch.Tensor,
                                dx: Optional[torch.Tensor] = None,
                                dx_index: Optional[torch.Tensor] = None):

    """
    Compute the writhe (signed crossing) of 2 segment pairs for NSegments and all frames (index 0) in xyz
     (xyz can contain just one frame)

    This computation uses cross products, which are the most precise for small angles.


    Args:
        segments:  tensor of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
                 tensor(Nframes, [seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point]).
                 We have the flexability for segments to simply be one dimensional if only one value of writhe is to be computed.

        xyz: tensor of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons


        We provide a way to use precomputed vectors when they're available with the caveat that the direction
        of displacements and the construction of the correct indices must follow the same conventions as if they
        were computed on the fly. This is considerably convoluted, so there's almost never a reason to use these arguments
        outside the context where graph objects containing the displacement vectors are already available.

        WARNING : precomputing and implementing fancy indexing may not increase speed due to resolution of addresses,
                  scattered reads, and non-contiguous sections of precomuted tensors being accessed in a single slice.


        dx : tensor of shape (***, 3), displacement vectors between C-alpha atoms.
         The exact form depends on dx_indices (if applicable, see WritheMessage class for a use-case)

        dx_index : tensor of shape (Nsegments, 6).
         The indices of the 6 displacement vectors in dx used to compute the writhe of 1 crossing.
    """

    # ensure correct shape for segment for lazy arguments
    assert segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
    segments, xyz = segments.unsqueeze(0) if segments.ndim < 2 else segments, \
                    xyz.unsqueeze(0) if xyz.ndim < 3 else xyz

    if dx is None or dx_index is None:
        # providing just one introduces ambiguity that can lead to unexpected results,
        # because the directions should be consistent, i.e, from i to j or i to j but not both (sign flips)
        tri = torch.triu_indices(xyz.shape[1], xyz.shape[1], offset=1)
        # always point towards the larger index
        dx = divnorm(xyz[:, tri[1]] - xyz[:, tri[0]]) if dx is None else dx
        # compute the writhe directly from these vectors
        dx_index = torch.stack([flat_index(segments[:, i],
                                           segments[:, j],
                                           n=xyz.shape[1],
                                           d0=1,
                                           triu=True)
                                for i, j in zip((0, 0, 0, 1, 1, 2),
                                                (1, 2, 3, 2, 3, 3))], 1).long()

    dx = dx.unsqueeze(0) if dx.ndim < 3 else dx

    crosses = divnorm(torch.cross(dx[:, dx_index[:, [1, 2, 4, 3]]],
                                  dx[:, dx_index[:, [2, 4, 3, 1]]],
                                  dim=-1)
                      )
    # gets crosses out of memory, this is actually the sum of the angles - 2 * pi
    crosses = (crosses[:, :, [0, 1, 2, 3]]
               * crosses[:, :, [1, 2, 3, 0]]
               ).sum(-1).clip(-1, 1).arcsin().sum(2)

    signs = (torch.cross(dx[:, dx_index[:, 5]],
                         dx[:, dx_index[:, 0]],
                         dim=-1) * dx[:, dx_index[:, 1]]).sum(-1).sign()

    return torch.squeeze(crosses * signs) / (2 * torch.pi)


@torch.jit.script
def writhe_segments_dot(xyz: torch.Tensor,
                        segments: torch.Tensor,
                        double: bool = False):
    """
    Compute the writhe (signed crossing) of 2 segment pairs for NSegments and all frames (index 0) in xyz
    (xyz can contain just one frame).

    This computation trades speed for precision. In cases where very small angles are encountered,
    this approach has more error than the methods using cross products. However,
    it's worth noting that this formulation is analytically correct and that the issues with precision
    stem from computing magnitudes of cross products and determinants with the gram determinant
    rather than scalar triple products.

    When using double precision, this approach gives results that are agree with the cross product method up to ~ 1e-5

    **Arguments**:
    - segments: Array of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:
      array(Nframes, [seg1_start_point, seg1_end_point, seg2_start_point, seg2_end_point]).
      Segments can be one-dimensional if only one value of writhe is to be computed.
    - xyz: Array of shape (Nframes, N_alpha_carbons, 3), coordinate array giving the positions of all alpha carbons.

    The program uses dot products and trigonometric identities to refactor the computation, making it
    ~twice as fast as the cross product formulation.

    It makes use of the following identities :

    Assume that a, b, c, d are normalized vectors in R^3:

         a.cross(b) · c.cross(d)  /  ||a.cross(b)|| * ||c.cross(d)||


        = [(a · c) * (b · d) - (a · d) * (b · c)] / sqrt( (1 - (a · b)**2) * (1 - (c · d)**2) )

    In the special case where b == c:

        = [(a · b) * (b · d) - (a · d) * ||b||**2 ] / sqrt( (1 - (a · b)**2) * (1 - (b · d)**2) )

    Note that ||·||**2 == 1.

    Set
    u = (a · b),
    v = (b · d),
    h = (a · d)
    and define the scalar function:

       sin(theta - pi / 2) = (u * v - h) / sqrt( (1 - u**2) * (1 - v**2) )

    Precomputing dot products and recursive application of this formula to compute the
    interior angles of the spherical quadrilateral allows computation of the writhe
    with 6 dot products and a determinant per crossing.
    """
    # ensure shape for segments
    assert segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
    dtype = xyz.dtype
    # catch any lazy arguments
    segments, xyz = segments.unsqueeze(0) if segments.ndim < 2 else segments, \
                     xyz.unsqueeze(0) if xyz.ndim < 3 else xyz
    xyz = xyz.double() if double else xyz

    # displacement vectors between end points
    dx = divnorm((-xyz[:, segments[:, :2], None] + xyz[:, segments[:, None, 2:]]
                 ).reshape(-1, len(segments), 4, 3))

    # compute all dot products, then work with scalars
    dots = (dx[:, :, [0, 0, 0, 1, 1, 2]] * dx[:, :, [1, 2, 3, 2, 3, 3]]).sum(-1)

    # get indices; dots is ordered according to indices of 3,3 upper right triangle
    # NOTE : we compute dots first because we use each twice
    u, v, h = [0, 4, 5, 1], \
              [4, 5, 1, 0], \
              [2, 3, 2, 3]

    # surface area from scalars
    dots = ((dots[:, :, u] * dots[:, :, v] - dots[:, :, h])

             / ((1 - dots[:, :, u] ** 2) * (1 - dots[:, :, v] ** 2)).clamp(min=1e-15).sqrt()

             ).clip(-1, 1).arcsin().sum(-1)

    # signs from triple products
    signs = (torch.cross(xyz[:, segments[:, 3]] - xyz[:, segments[:, 2]],
                         xyz[:, segments[:, 1]] - xyz[:, segments[:, 0]],
                         dim=-1) * dx[:, :, 0]).sum(-1).sign()

    return torch.squeeze(dots * signs).to(dtype) / (2 * torch.pi)

@torch.jit.script
def writhe_segments_dot_index(xyz: torch.Tensor,
                              segments: torch.Tensor,
                              dx: Optional[torch.Tensor] = None,
                              dx_index: Optional[torch.Tensor] = None,
                              double: bool = False):
    """
    Compute the writhe (signed crossing) of 2 segment pairs for NSegments and all frames (index 0) in xyz
    (xyz can contain just one frame).

    This computation trades speed for precision. In cases where very small angles are encountered,
    this approach has more error than the methods using cross products. However,
    it's worth noting that this formulation is analytically correct and that the issues with precision
    stem from computing magnitudes of cross products and determinants with the gram determinant
    rather than scalar triple products.

    When using double precision, this approach gives results that are agree with the cross product method up to ~ 1e-5

    **Arguments**:
    - segments: Array of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:
      array(Nframes, [seg1_start_point, seg1_end_point, seg2_start_point, seg2_end_point]).
      Segments can be one-dimensional if only one value of writhe is to be computed.
    - xyz: Array of shape (Nframes, N_alpha_carbons, 3), coordinate array giving the positions of all alpha carbons.

    The program uses dot products and trigonometric identities to refactor the computation, making it
    ~twice as fast as the cross product formulation.

    It makes use of the following identities :

    Assume that a, b, c, d are normalized vectors in R^3:

         a.cross(b) · c.cross(d)  /  ||a.cross(b)|| * ||c.cross(d)||


        = [(a · c) * (b · d) - (a · d) * (b · c)] / sqrt( (1 - (a · b)**2) * (1 - (c · d)**2) )

    In the special case where b == c:

        = [(a · b) * (b · d) - (a · d) * ||b||**2 ] / sqrt( (1 - (a · b)**2) * (1 - (b · d)**2) )

    Note that ||·||**2 == 1.

    Set
    u = (a · b),
    v = (b · d),
    h = (a · d)
    and define the scalar function:

       sin(theta - pi / 2) = (u * v - h) / sqrt( (1 - u**2) * (1 - v**2) )

    Precomputing dot products and recursive application of this formula to compute the
    interior angles of the spherical quadrilateral allows computation of the writhe
    with 6 dot products and a determinant per crossing.
    """
    # ensure shape for segments
    assert segments.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
    dtype = xyz.dtype
    # catch any lazy arguments
    segments, xyz = segments.unsqueeze(0) if segments.ndim < 2 else segments, \
                     xyz.unsqueeze(0) if xyz.ndim < 3 else xyz
    xyz = xyz.double() if double else xyz

    if dx is None or dx_index is None:
        # providing just one introduces ambiguity that can lead to unexpected results,
        # because the directions should be consistent, i.e, from i to j or i to j but not both (sign flips)
        tri = torch.triu_indices(xyz.shape[1], xyz.shape[1], offset=1)
        # always point towards the larger index
        dx = divnorm(xyz[:, tri[1]] - xyz[:, tri[0]]) if dx is None else dx
        # compute the writhe directly from these vectors
        dx_index = torch.stack([flat_index(segments[:, i],
                                           segments[:, j],
                                           n=xyz.shape[1],
                                           d0=1,
                                           triu=True)
                                for i, j in zip((0, 0, 0, 1, 1, 2),
                                                (1, 2, 3, 2, 3, 3))], 1).long()

    dx = dx.unsqueeze(0) if dx.ndim < 3 else dx

    # compute all dot products, then work with scalars
    dots = (dx[:, dx_index[:, [1, 1, 1, 2, 2, 3]]] * dx[:, dx_index[:, [2, 3, 4, 3, 4, 4]]]).sum(-1)

    # get indices; dots is ordered according to indices of 3,3 upper right triangle
    # NOTE : we compute dots first because we use each twice
    u, v, h = [0, 4, 5, 1], \
              [4, 5, 1, 0], \
              [2, 3, 2, 3]

    # surface area from scalars
    dots = ((dots[:, :, u] * dots[:, :, v] - dots[:, :, h])

             / ((1 - dots[:, :, u] ** 2) * (1 - dots[:, :, v] ** 2)).clamp(min=1e-15).sqrt()

             ).clip(-1, 1).arcsin().sum(-1)

    # signs from triple products
    signs = (torch.cross(dx[:, dx_index[:, 5]],
                         dx[:, dx_index[:, 0]],
                         dim=-1) * dx[:, dx_index[:, 1]]).sum(-1).sign()

    return torch.squeeze(dots * signs).to(dtype) / (2 * torch.pi)


class WritheMessage(nn.Module):
    """
    Expedited Writhe message layer.

    """

    def __init__(self,
                 n_atoms: int,
                 n_features: int,
                 batch_size: int,
                 bins: int = 300,
                 node_attr: str = "invariant_node_features",
                 edge_attr: str = "invariant_edge_features",
                 distance_attr: str = "edge_dist",
                 distance_attention: bool = True,
                 #dir_attr: str = "edge_dir",
                 segment_length: int = 1,
                 gaussian_bins: bool = False,
                 bin_low: float = -1,
                 bin_high: float = 1,
                 bin_std: float = 1,
                 bin_mean: float = 0,
                 residual: bool = True
                 ):

        super().__init__()

        from torch_scatter import scatter

        # prerequisit information
        segments = get_segments(n_atoms, length=segment_length, tensor=True)
        # need edges based on the segments the writhe is computed from
        edges = torch.cat([segments[:, [0, 2]], segments[:, [1, 3]]]).T
        # edges = torch.cat([edges, torch.flip(edges, (0,))], dim=1) # this doesn't help


            # self.register_buffer("distance_indices",
            #                      distance_indices)
            # self.register_buffer("distance_indices",
            #                      torch.cat([i * increment + dist_index for i in range(batch_size)]))
        #else:

        # if dir_attr is not None:
        #     dx_index = torch.stack([flat_index(segments[:, i],
        #                                        segments[:, j],
        #                                        n=n_atoms,
        #                                        d0=1,
        #                                        triu=False)
        #                             for i, j in zip((0, 0, 0, 1, 1, 2),
        #                                             (1, 2, 3, 2, 3, 3))], 1).long()
        #     # retrieve the needed displacement vectors
        #     # in the graph object (if they're there)
        #     self.register_buffer("dir_indices",
        #                          (dx_index.repeat(batch_size, 1, 1)
        #                           + int(n_atoms ** 2 - n_atoms)
        #                           * torch.arange(batch_size).reshape(-1, 1, 1)
        #                           ).flatten(0, 1)
        #                          )
        #     #self.register_buffer("dir_signs")
        #     # the writhe has to be computed with a fixed sign convention
        #     # signs

        self.register_buffer("edge_indices",
                            (flat_index(*edges, n=n_atoms, d0=1).repeat(batch_size, 1)
                            + int(n_atoms ** 2 - n_atoms) * torch.arange(batch_size).unsqueeze(-1)
                            ).flatten()
                             )

        self.register_buffer("edges",
                             torch.cat([i * n_atoms + edges for i in range(batch_size)], axis=1).long())

        self.register_buffer("segments", segments)
        self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
        self.register_buffer("batch_size_", torch.LongTensor([batch_size]))
        self.register_buffer("segment_length", torch.LongTensor([segment_length]))
        self.register_buffer("distance_attention_", torch.LongTensor([int(distance_attention)]))

        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.distance_attr = distance_attr
        #self.dir_attr = dir_attr
        self.n_features = n_features
        self.residual = residual
        self.bin_low = bin_low
        self.bin_high = bin_high

        # writhe embedding

        self.soft_one_hot = GaussEncoder(low=self.bin_low,
                                         high=self.bin_high,
                                         number=bins,
                                         gauss=gaussian_bins,
                                         std=bin_std,
                                         mu=bin_mean)

        std = 1. / math.sqrt(n_features)  # 2 / bins,  this is how torch initializes embedding layers

        self.register_parameter("basis",
                                torch.nn.Parameter(
                                    torch.Tensor(1, 1, bins, n_features
                                                 ).uniform_(-std, std),  # normal_(0, std)
                                    requires_grad=True)
                                )
        if not distance_attention:
            self.attention = nn.Sequential(nn.Linear(n_features * 3, n_features, bias=False),
                                           nn.SiLU(),
                                           nn.Linear(n_features, 1, bias=False)
                                           )


    @property
    def n_atoms(self):
        return self.n_atoms_.item()

    @property
    def distance_attention(self):
        return self.distance_attention_.item()


    def embed_writhe(self, wr):
        return (self.soft_one_hot(wr).unsqueeze(-1) * self.basis).sum(-2)

    def compute_writhe(self, x):
        # if hasattr(x, str(self.dir_attr)):
        #     i, j = (i.flatten() for i in x.edge_index)
        #     # the writhe computation requires a fixed direction for displacement vectors,
        #     # i.e. i -> j or j -> i but not both as the same computation will give opposite signs
        #     dx, dx_index = divnorm(x[self.dir_attr] * (i - j).sign().reshape(-1, 1)), self.dir_indices
        # else:
        #     dx, dx_index = None, None

        return self.embed_writhe(
            writhe_segments_cross(
                xyz=x.x.reshape(-1, self.n_atoms, 3),
                segments=self.segments,
                # dx=dx,
                # dx_index=dx_index
               )).repeat(1, 2, 1).reshape(-1, self.n_features)

    def forward(self, x, update=True):

        src_node, dst_node = (i.flatten() for i in self.edges)
        writhe = self.compute_writhe(x)
        # derive attention weights from distances - unrelated to calculation of the writhe from coordinates
        # if no distances computed, compute them
        if self.distance_attention:

            weights = (getattr(x, self.distance_attr)[self.distance_indices]
                   if hasattr(x, str(self.distance_attr))
                   else (x.x[src_node] - x.x[dst_node]).norm(dim=-1)
                   ).pow(2).neg().exp()
        else:
            weights = self.attention(
                torch.cat([x[self.node_attr][src_node],
                           x[self.node_attr][dst_node],
                           x[self.edge_attr][self.edge_indices]\
                               if hasattr(x, str(self.edge_attr)) else writhe
                           ], 1))

        # get attention weights, softmax normalize, scatter
        attention = (weights / scatter(weights, dst_node)[dst_node]).unsqueeze(-1)
        message = scatter(writhe * attention, dst_node, dim=0)

        if update:
            x[self.node_attr] = x[self.node_attr] + message if self.residual else message
            return x
        else:
            return x[self.node_attr] + message if self.residual else message


class GaussEncoder(nn.Module):
    def __init__(self,
                 low: float,
                 high: float,
                 number: int,
                 gauss: bool = False,
                 std: float = 1,
                 mu: float = 0):

        super().__init__()

        if gauss:
            self.register_buffer("step_", torch.diff(gaussian_binning(low, high, number, std, mu)))
            self.register_buffer("bin_centers", gaussian_binning(low, high, number - 1, std, mu))

        else:
            bins = torch.linspace(low, high, number)
            self.register_buffer("step_", torch.Tensor([bins[1] - bins[0]]))
            self.register_buffer("bin_centers", bins)

    @property
    def step(self):
        return self.step_.item() if len(self.step_) == 1 else self.step_

    def forward(self, x):

        diff = (x[..., None] - self.bin_centers) / self.step

        return diff.pow(2).neg().exp().div(1.12)


def gaussian_binning(low: float,
                     high: float,
                     number: int,
                     std: float = 1,
                     mean: float = 0):
    """
    Creates symmetric non-uniform bins where bin
    widths are proportional to the height of a Gaussian distribution.

    Parameters:
    - low: low of the interval.
    - high: high of the interval.
    - number: Number of bins.
    - std: Standard deviation of the Gaussian distribution.
    - mean: Mean of the Gaussian distribution.

    Returns:
    - bin_edges: Tensor of bin edges (length: num_bins + 1), symmetric and in increasing order.
    """
    number += 2
    # Generate a Gaussian PDF over the specified symmetric range
    x = torch.linspace(low, high, 500000)  # we only compute this once (yee haw)
    pdf = torch.distributions.Normal(loc=mean, scale=std).log_prob(x).exp()
    # Normalize the PDF so that it sums to 1
    pdf /= torch.sum(pdf)
    # Create a cumulative sum to represent the integral of the PDF
    pdf_cumsum = torch.cumsum(pdf, dim=0)
    # Target cumulative values for bin edges (0 to 1) evenly spaced
    target_cdf_vals = torch.linspace(0 - 1 / number,
                                     1 + 1 / number,
                                     number + 1)

    # Find where the target CDF values map onto the cumulative PDF
    indices = torch.searchsorted(pdf_cumsum, target_cdf_vals).clamp(0, len(x) - 1)

    # Linear interpolation for bin edge determination
    x0 = x[indices - 1].clamp(min=low)  # Lower bound for interpolation
    x1 = x[indices].clamp(max=high)  # Upper bound for interpolation
    y0 = pdf_cumsum[indices - 1].clamp(min=0)  # Corresponding CDF values
    y1 = pdf_cumsum[indices].clamp(max=1)  # Next CDF values

    # Linear interpolation formula for bin edges
    bin_edges = x0 + (target_cdf_vals - y0) * (x1 - x0) / (y1 - y0 + 1e-9)

    # To create symmetric bin edges about the mean
    left_half = mean - (bin_edges - mean)  # Reflect bin edges about the mean
    full_bin_edges = torch.cat((left_half.flip(0), bin_edges))

    # Ensure we only take 'number + 1' edges and sort them
    full_bin_edges = full_bin_edges[:number]
    final_bin_edges = torch.sort(full_bin_edges)[0]

    return final_bin_edges[1:]


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
