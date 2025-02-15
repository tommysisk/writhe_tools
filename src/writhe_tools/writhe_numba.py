import numba
import numpy as np
from numba import prange


@numba.njit(fastmath=True, parallel=True)
def norm(x):
    """Compute the Euclidean norm along the last axis."""
    return np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2).reshape(-1, 1)

@numba.njit(fastmath=True, parallel=True)
def divnorm(x):
    """Normalize an (n,3) array along the last axis."""
    n = norm(x)
    return x / n

@numba.njit(fastmath=True, parallel=True)
def diffnorm(x, y):
    """Compute the normalized difference between two (n,3) arrays."""
    dx = x - y
    return dx / norm(dx)

@numba.njit(fastmath=True)
def displacements(xyz, segment):
    """Compute the displacements between segment endpoints."""
    n = xyz.shape[0]
    dx = np.empty((n, 4, 3))  # Use np.empty() to avoid redundant initialization
    s1, s2, s3, s4 = segment[0], segment[1], segment[2], segment[3]
    dx[:, 0] = diffnorm(xyz[:, s3], xyz[:, s1])
    dx[:, 1] = diffnorm(xyz[:, s4], xyz[:, s1])
    dx[:, 2] = diffnorm(xyz[:, s3], xyz[:, s2])
    dx[:, 3] = diffnorm(xyz[:, s4], xyz[:, s2])
    return dx

@numba.njit(fastmath=True, parallel=True)
def dot(a, b):
    """Compute the dot product along the last axis."""
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

@numba.njit(fastmath=True, parallel=True)
def cross(a, b):
    """Compute the cross product along the last axis."""
    result = np.empty((a.shape[0], 3), dtype=a.dtype)
    result[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    result[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    result[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return result

@numba.njit(fastmath=True, parallel=True)
def crosses(dx, idx):
    """Compute normalized cross products for an index list."""
    c = np.empty((dx.shape[0], len(idx), dx.shape[-1]))  # Avoid np.zeros() overhead
    for i in prange(len(idx)):
        c[:, i] = divnorm(cross(dx[:, idx[i, 0]], dx[:, idx[i, 1]]))
    return c

@numba.njit(fastmath=True, parallel=True)
def dots(dx, idx):
    """Compute dot products for an index list."""
    d = np.empty((dx.shape[0], len(idx)))  # Use np.empty() instead of np.zeros()
    for i in prange(len(idx)):
        d[:, i] = dot(dx[:, idx[i, 0]], dx[:, idx[i, 1]])
    return d


@numba.njit(fastmath=True)
def writhe_segment(xyz, segment, use_cross=True):
    xyz = xyz[None, :] if xyz.ndim < 3 else xyz

    # Extract segment indices explicitly (Fixes unsupported multi-dimensional indexing)
    s1, s2, s3, s4 = segment[0], segment[1], segment[2], segment[3]

    # Compute displacements manually
    dx = np.empty((xyz.shape[0], 4, 3))  # Pre-allocate memory
    dx[:, 0] = divnorm(xyz[:, s3] - xyz[:, s1])
    dx[:, 1] = divnorm(xyz[:, s4] - xyz[:, s1])
    dx[:, 2] = divnorm(xyz[:, s3] - xyz[:, s2])
    dx[:, 3] = divnorm(xyz[:, s4] - xyz[:, s2])

    if use_cross:
        theta = np.empty((xyz.shape[0], 4, 3))  # Pre-allocate theta storage
        theta[:, 0] = divnorm(cross(dx[:, 0], dx[:, 1]))
        theta[:, 1] = divnorm(cross(dx[:, 1], dx[:, 3]))
        theta[:, 2] = divnorm(cross(dx[:, 3], dx[:, 2]))
        theta[:, 3] = divnorm(cross(dx[:, 2], dx[:, 0]))

        # Compute sum over arcsin terms
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

        # Compute arcsin terms
        u, v, h = [0, 4, 5, 1], [4, 5, 1, 0], [2, 3, 2, 3]
        theta_sum = np.zeros(xyz.shape[0])

        for i in range(4):
            theta_sum += np.arcsin(((dot_theta[:, u[i]] * dot_theta[:, v[i]] - dot_theta[:, h[i]]) /
                                   np.sqrt(np.abs((1 - dot_theta[:, u[i]] ** 2) *
                                                  (1 - dot_theta[:, v[i]] ** 2)).clip(1e-17))
                                   ).clip(-1, 1))

    # Compute signs
    sign_cross = cross(xyz[:, s4] - xyz[:, s3], xyz[:, s2] - xyz[:, s1])
    signs = np.sign(dot(dx[:, 0], sign_cross))

    return (theta_sum * signs) / (2 * np.pi)

@numba.njit(fastmath=True, parallel=True)
def writhe_segments_numba(xyz, segments, use_cross=True):
    """Compute the writhe contributions of multiple segments."""
    writhe = np.empty((len(xyz), len(segments)))  # Use np.empty() instead of np.zeros()
    for i in prange(len(segments)):  # Efficient parallel processing
        writhe[:, i] = writhe_segment(xyz, segments[i], use_cross)
    return writhe



