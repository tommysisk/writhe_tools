
import torch
import torch.nn as nn
import numpy as np
import ray
from .utils import split_list, shifted_pairs, product, combinations
from .writhe_nn import ndot, ncross, nnorm, writhe_segments


# how to compute writhe angle (omega) using ortho projection rather than cross products. Still appears to be slower than cross products.
# smat = xyz[:, segments]
# n_segments = smat.shape[1]
# indices = torch.LongTensor([[0, 1, 0, 2],
#                             [3, 2, 3, 1],
#                             [1, 3, 2, 0]
#                            ]).T
#
# dx = nnorm((-smat[:, :, :2, None, :] + smat[:, :, None, 2:, :]).reshape(-1, n_segments, 4, 3)) #displacements
# omega = (-torch.arcsin(uproj(dx[:, :, indices[:,:-1]], dx[:, :, indices[:, -1], None]).prod(dim=-2).sum(-1).clip(-1, 1))).sum(-1)

def get_segments(n: int = None,
                 length: int = 1,
                 index0: np.ndarray = None,
                 index1: np.ndarray = None,
                 tensor: bool = False):
    """
    Function to retrieve indices of segment pairs for various use cases.
    Returns an (n_segment_pairs, 4) array where each row (quadruplet) contains : (start1, end1, start2, end2)
    """

    if all(i is None for i in (index0, index1)):
        assert n is not None, \
            "Must provide indices (index0:array, (optionally) index1:array) or the number of points (n: int)"
        segments = combinations(shifted_pairs(np.arange(n), length)).reshape(-1, 4)
        segments = segments[~(segments[:, 1] == segments[:, 2])]
        return torch.from_numpy(segments).long() if tensor else segments

    else:
        assert index0 is not None, ("If providing only one set of indices, must set the index0 argument \n"
                                    "Cannot only supply the index1 argument (doesn't make sense in this context")
        if index1 is not None:
            segments = product(*[shifted_pairs(i, length) for i in (index0, index1)]).reshape(-1, 4)
            return torch.from_numpy(segments).long() if tensor else segments
        else:
            segments = combinations(shifted_pairs(index0, length)).reshape(-1, 4)
            segments = segments[~(segments[:, 1] == segments[:, 2])]
            return torch.from_numpy(segments).long() if tensor else segments


class TorchWrithe(nn.Module):
    """
    Class to compute writhe using torch
    """

    def __init__(self,
                 n_atoms: int,
                 segment_length: int = 1,
                 ):
        super().__init__()

        self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
        self.segment_length = segment_length

    @property
    def n_atoms(self):
        return self.n_atoms_.item()

    @property
    def segment_length(self):
        return self.segment_length_

    @segment_length.setter
    def segment_length(self, length: int):
        self.segment_length_ = length
        self.register_buffer("segments", get_segments(self.n_atoms, length, tensor=True))
        return

    def forward(self, xyz: torch.Tensor):
        return writhe_segments(self.segments, xyz)


# for computing but not deep learning, attempts to avoid leaving tensors hiding on GPUs
def writhe_smat(smat: torch.Tensor, device: int = 0):
    """
    Function to use GPU in computation of the writhe; NOT to be used in neural nets
    smat: array of shape(Nframes, Nsegments, 4, 3), coordinate array sliced with segments array

    """
    smat = smat.to(device)

    displacements = nnorm((-smat[:, :, :2, None, :] + smat[:, :, None, 2:, :]
                           ).reshape(-1, smat.shape[1], 4, 3))

    signs = torch.sign(ndot(ncross(smat[:, :, 3] - smat[:, :, 2],
                                   smat[:, :, 1] - smat[:, :, 0]),
                            displacements[:, :, 0]))

    del smat

    crosses = nnorm(ncross(displacements[:, :, [0, 1, 3, 2]], displacements[:, :, [1, 3, 2, 0]]))

    del displacements

    omega = torch.arcsin(ndot(crosses[:, :, [0, 1, 2, 3]],
                              crosses[:, :, [1, 2, 3, 0]]).clip(-1, 1)).sum(2)

    del crosses

    return (omega * signs / (2 * torch.pi)).cpu()


def peak_mem_writhe(n_segments, n_samples):
    """
    Return the estimated peak memory required by a broadcasted writhe calculation using
    n_segments and n_samples in GB.

    This was determined through numerical anaylsis, however, can only be considered a rule of thumb.

    Torch seems to need about 2 GB more than expected on some machines.
    """
    return (n_segments * 2.01708461e-07 + 5.93515514e-08) * n_samples


def get_available_memory(device: int = 0):
    """return VRAM available on a device in GB"""

    assert torch.cuda.is_available(), "CUDA is not available"

    return (torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)) / 1024 ** 3


def get_segment_batch_size(n_samples, device: int = 0):
    mem = get_available_memory() - 2  # torch seems to need 2 GB more than expected

    return int(((mem / n_samples) - 5.93515514e-08) / 2.01708461e-07)


def writhe_segments_cuda(segments, xyz, device: int = 0):
    segments = segments.unsqueeze(0) if segments.ndim < 2 else segments
    smat = (xyz.unsqueeze(0) if xyz.ndim < 3 else xyz)[:, segments]
    result = writhe_smat(smat=smat, device=device)
    torch.cuda.empty_cache()
    return result


def writhe_segments_minibatches(segment_chunks: list, xyz: torch.tensor, device: int):
    xyz = ray.get(xyz) if not isinstance(xyz, torch.Tensor) else xyz
    return torch.cat([writhe_segments_cuda(segments=i, xyz=xyz, device=device) for i in segment_chunks], -1)


def calc_writhe_parallel_cuda(segments: torch.Tensor,
                              xyz: torch.Tensor,
                              reduce_batch_size: int = 0):
    batch_size = get_segment_batch_size(len(xyz)) - reduce_batch_size

    if batch_size > len(segments):
        return writhe_segments_cuda(segments, xyz).numpy()

    chunks = list(torch.split(segments, batch_size))

    if len(segments) < 5 * batch_size or torch.cuda.device_count() == 1:
        return writhe_segments_minibatches(chunks, xyz, 0).numpy()

    else:
        minibatches = split_list(chunks, torch.cuda.device_count())
        xyz_ref = ray.put(xyz)
        fxn = ray.remote(num_gpus=torch.cuda.device_count())(writhe_segments_minibatches)
        return torch.cat(ray.get([fxn.remote(segment_chunks=j, xyz=xyz_ref, device=i)
                                  for i, j in enumerate(minibatches)]), -1).numpy()
