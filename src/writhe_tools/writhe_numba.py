from numba import prange
import numpy as np
import numba

import numpy as np
import numba
from numba import prange

@numba.njit
def pbc_correct(x, lengths):
    N = x.shape[0]
    ndim = x.ndim
    out = np.empty_like(x)

    if ndim == 2:
        for i in range(N):
            box = lengths[i]
            out[i] = x[i] - box * np.round(x[i] / box)
    elif ndim == 3:
        for i in range(N):
            box = lengths[i]
            for j in range(x.shape[1]):
                out[i, j] = x[i, j] - box * np.round(x[i, j] / box)
    elif ndim == 4:
        for i in range(N):
            box = lengths[i]
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    out[i, j, k] = x[i, j, k] - box * np.round(x[i, j, k] / box)
    else:
        raise ValueError("Input must have shape (N, ..., 3) with up to 3 trailing dims.")
    return out

@numba.njit(fastmath=True)
def norm(x):
    return np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2).reshape(-1, 1)

@numba.njit(fastmath=True)
def divnorm(x):
    n = norm(x)
    return x / n

@numba.njit
def diffnorm(x, y):
    dx = x - y
    return dx / norm(dx)

@numba.njit
def diffnorm_pbc(x, y, lengths):
    dx = x - y
    dx -= lengths * np.round(dx / lengths)
    return dx / norm(dx)

@numba.njit(fastmath=True)
def displacements(xyz, segment):
    n = xyz.shape[0]
    dx = np.empty((n, 4, 3))
    s1, s2, s3, s4 = segment[0], segment[1], segment[2], segment[3]
    dx[:, 0] = diffnorm(xyz[:, s3], xyz[:, s1])
    dx[:, 1] = diffnorm(xyz[:, s4], xyz[:, s1])
    dx[:, 2] = diffnorm(xyz[:, s3], xyz[:, s2])
    dx[:, 3] = diffnorm(xyz[:, s4], xyz[:, s2])
    return dx

@numba.njit(fastmath=True)
def displacements_pbc(xyz, segment, lengths):
    n = xyz.shape[0]
    dx = np.empty((n, 4, 3))
    s1, s2, s3, s4 = segment[0], segment[1], segment[2], segment[3]
    dx[:, 0] = diffnorm_pbc(xyz[:, s3], xyz[:, s1], lengths)
    dx[:, 1] = diffnorm_pbc(xyz[:, s4], xyz[:, s1], lengths)
    dx[:, 2] = diffnorm_pbc(xyz[:, s3], xyz[:, s2], lengths)
    dx[:, 3] = diffnorm_pbc(xyz[:, s4], xyz[:, s2], lengths)
    return dx

@numba.njit(fastmath=True)
def dot(a, b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

@numba.njit(fastmath=True)
def cross(a, b):
    result = np.empty((a.shape[0], 3), dtype=a.dtype)
    result[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    result[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    result[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return result

@numba.njit(fastmath=True)
def writhe_segment(xyz, segment, use_cross=True):
    xyz = xyz[None, :] if xyz.ndim < 3 else xyz
    s1, s2, s3, s4 = segment[0], segment[1], segment[2], segment[3]
    dx = displacements(xyz, segment)
    sign_cross = cross(xyz[:, s4] - xyz[:, s3], xyz[:, s2] - xyz[:, s1])

    if use_cross:
        theta = np.empty((xyz.shape[0], 4, 3))
        theta[:, 0] = divnorm(cross(dx[:, 0], dx[:, 1]))
        theta[:, 1] = divnorm(cross(dx[:, 1], dx[:, 3]))
        theta[:, 2] = divnorm(cross(dx[:, 3], dx[:, 2]))
        theta[:, 3] = divnorm(cross(dx[:, 2], dx[:, 0]))

        arc_theta = np.empty((xyz.shape[0], 4))
        arc_theta[:, 0] = np.arcsin(np.clip(dot(theta[:, 0], theta[:, 1]), -1, 1))
        arc_theta[:, 1] = np.arcsin(np.clip(dot(theta[:, 1], theta[:, 2]), -1, 1))
        arc_theta[:, 2] = np.arcsin(np.clip(dot(theta[:, 2], theta[:, 3]), -1, 1))
        arc_theta[:, 3] = np.arcsin(np.clip(dot(theta[:, 3], theta[:, 0]), -1, 1))

        theta_sum = arc_theta.sum(1)
    else:
        dot_theta = np.empty((xyz.shape[0], 6))
        dot_theta[:, 0] = dot(dx[:, 0], dx[:, 1])
        dot_theta[:, 1] = dot(dx[:, 0], dx[:, 2])
        dot_theta[:, 2] = dot(dx[:, 0], dx[:, 3])
        dot_theta[:, 3] = dot(dx[:, 1], dx[:, 2])
        dot_theta[:, 4] = dot(dx[:, 1], dx[:, 3])
        dot_theta[:, 5] = dot(dx[:, 2], dx[:, 3])

        u, v, h = [0, 4, 5, 1], [4, 5, 1, 0], [2, 3, 2, 3]
        theta_sum = np.zeros(xyz.shape[0])
        for i in range(4):
            theta_sum += np.arcsin(((dot_theta[:, u[i]] * dot_theta[:, v[i]] - dot_theta[:, h[i]]) /
                                    np.sqrt(np.abs((1 - dot_theta[:, u[i]] ** 2) *
                                                   (1 - dot_theta[:, v[i]] ** 2)).clip(1e-17))).clip(-1, 1))

    signs = np.sign(dot(dx[:, 0], sign_cross))
    return (theta_sum * signs) / (2 * np.pi)

@numba.njit(fastmath=True)
def writhe_segment_pbc(xyz, segment, lengths, use_cross=True):
    xyz = xyz[None, :] if xyz.ndim < 3 else xyz
    s1, s2, s3, s4 = segment[0], segment[1], segment[2], segment[3]
    dx = displacements_pbc(xyz, segment, lengths)
    sign_cross = cross(pbc_correct(xyz[:, s4] - xyz[:, s3], lengths),
                       pbc_correct(xyz[:, s2] - xyz[:, s1], lengths))

    if use_cross:
        theta = np.empty((xyz.shape[0], 4, 3))
        theta[:, 0] = divnorm(cross(dx[:, 0], dx[:, 1]))
        theta[:, 1] = divnorm(cross(dx[:, 1], dx[:, 3]))
        theta[:, 2] = divnorm(cross(dx[:, 3], dx[:, 2]))
        theta[:, 3] = divnorm(cross(dx[:, 2], dx[:, 0]))

        arc_theta = np.empty((xyz.shape[0], 4))
        arc_theta[:, 0] = np.arcsin(np.clip(dot(theta[:, 0], theta[:, 1]), -1, 1))
        arc_theta[:, 1] = np.arcsin(np.clip(dot(theta[:, 1], theta[:, 2]), -1, 1))
        arc_theta[:, 2] = np.arcsin(np.clip(dot(theta[:, 2], theta[:, 3]), -1, 1))
        arc_theta[:, 3] = np.arcsin(np.clip(dot(theta[:, 3], theta[:, 0]), -1, 1))

        theta_sum = arc_theta.sum(1)
    else:
        dot_theta = np.empty((xyz.shape[0], 6))
        dot_theta[:, 0] = dot(dx[:, 0], dx[:, 1])
        dot_theta[:, 1] = dot(dx[:, 0], dx[:, 2])
        dot_theta[:, 2] = dot(dx[:, 0], dx[:, 3])
        dot_theta[:, 3] = dot(dx[:, 1], dx[:, 2])
        dot_theta[:, 4] = dot(dx[:, 1], dx[:, 3])
        dot_theta[:, 5] = dot(dx[:, 2], dx[:, 3])

        u, v, h = [0, 4, 5, 1], [4, 5, 1, 0], [2, 3, 2, 3]
        theta_sum = np.zeros(xyz.shape[0])
        for i in range(4):
            theta_sum += np.arcsin(((dot_theta[:, u[i]] * dot_theta[:, v[i]] - dot_theta[:, h[i]]) /
                                    np.sqrt(np.abs((1 - dot_theta[:, u[i]] ** 2) *
                                                   (1 - dot_theta[:, v[i]] ** 2)).clip(1e-17))).clip(-1, 1))

    signs = np.sign(dot(dx[:, 0], sign_cross))
    return (theta_sum * signs) / (2 * np.pi)

@numba.njit(fastmath=True, parallel=True)
def writhe_segments_numba(xyz, segments, use_cross=True):
    writhe = np.empty((len(xyz), len(segments)))
    for i in prange(len(segments)):
        writhe[:, i] = writhe_segment(xyz, segments[i], use_cross)
    return writhe

@numba.njit(fastmath=True, parallel=True)
def writhe_segments_numba_pbc(xyz, segments, lengths, use_cross=True):
    writhe = np.empty((len(xyz), len(segments)))
    for i in prange(len(segments)):
        writhe[:, i] = writhe_segment_pbc(xyz, segments[i], lengths, use_cross)
    return writhe


