import ray
import numpy as np
import functools
import multiprocessing


def divnorm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def writhe_segment(segment=None,
                   xyz=None,
                   use_cross: bool = True):
    """
    Version of the writhe computation written specifically for CPU + ray parallelization.
    See writhe_tools.writhe_nn.writhe_segments for cleaner implementation that is generally more efficient.
    We use this version because ray manages memory and parallelization for large computations.

    Compute the writhe (signed crossing) of 2 segments for all frames (index 0) in xyz (xyz can contain just one frame)
    **provide both of the following**
    segment: numpy array of shape (4,) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
             array([seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point])
    xyz: numpy array of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons

    """
    xyz = xyz[None, :] if xyz.ndim < 3 else xyz
    # broadcasting trick
    # negative sign, None placement and order are intentional, don't change without testing equivalent option
    dx = divnorm((-xyz[:, segment[:2], None] + xyz[:, segment[None, 2:]]).reshape(-1, 4, 3))
    signs = np.sign(np.sum(dx[:, 0] * np.cross(xyz[:, segment[3]] - xyz[:, segment[2]],
                                               xyz[:, segment[1]] - xyz[:, segment[0]], axis=-1),
                           axis=-1)).squeeze()
    # for the following, array broadcasting is (surprisingly) slower than list comprehensions
    # when using ray!! (without ray, broadcasting is faster).
    if use_cross:
        dx = np.stack([divnorm(np.cross(dx[:, i], dx[:, j], axis=-1))
                          for i, j in zip([0, 1, 3, 2], [1, 3, 2, 0])], axis=1)

        dx = np.stack([np.arcsin((dx[:, i] * dx[:, j]).sum(-1).clip(-1, 1))
                          for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0])], axis=1).squeeze().sum(1)

    else:
        dx = np.stack([(dx[:, i] * dx[:, j]).sum(-1)
                          for i, j in zip([0, 0, 0, 1, 1, 2],
                                          [1, 2, 3, 2, 3, 3])], axis=-1)
        # indices
        u, v, h = [0, 4, 5, 1], \
                  [4, 5, 1, 0], \
                  [2, 3, 2, 3]

        # surface area from scalars

        dx = np.sum([np.arcsin(((dx[:, i] * dx[:, j] - dx[:, k])
                                   / np.sqrt(abs(((1 - dx[:, i] ** 2) * (1 - dx[:, j] ** 2))).clip(1e-17))
                                   ).clip(-1, 1)) for i, j, k in zip(u, v, h)], axis=0)

    return (dx * signs) / (2 * np.pi)


# @ray.remote
def writhe_segments_along_axis(xyz: np.ndarray,
                               segments: np.ndarray,
                               use_cross: bool = True,
                               axis: int = 1):
    """helper function for parallelization to compute writhe over chuncks of segments for all frames in xyz"""
    # noinspection PyTypeChecker
    return np.apply_along_axis(func1d=functools.partial(writhe_segment,
                                                        xyz=xyz,
                                                        use_cross=use_cross
                                                        ),
                               axis=axis, arr=segments)


# Ray parallelization is substantially faster than python multiprocessing
def writhe_segments_ray(xyz: np.ndarray,
                        segments: np.ndarray,
                         use_cross: bool = True,
                         cpus_per_job: int = 1) -> "Nframes by Nsegments np.ndarray":
    """parallelize writhe calculation by segment, avoids making multiple copies of coordinate (xyz) matrix using Ray shared memory"""
    # ray.init()

    xyz_ref = ray.put(xyz)  # reference to coordinates that all child processes can access without copying
    writhe_segments_along_axis_ref = ray.remote(writhe_segments_along_axis)
    chunks = np.array_split(segments, int(multiprocessing.cpu_count() / cpus_per_job))
    result = np.concatenate(ray.get([writhe_segments_along_axis_ref.remote(segments=chunk,
                                                                           xyz=xyz_ref,
                                                                           use_cross=use_cross)
                                     for chunk in chunks])).T.squeeze()
    ray.internal.free(xyz_ref)
    ray.shutdown()
    return result
