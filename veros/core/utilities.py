from veros.core.operators import numpy as np

from veros import veros_kernel
from veros.core.operators import update, at, solve_tridiagonal


@veros_kernel(static_args=("enable_cyclic_x", "local"))
def enforce_boundaries(arr, enable_cyclic_x, local=False):
    from .. import runtime_settings as rs
    from ..routines import CURRENT_CONTEXT

    if rs.num_proc[0] == 1 or not CURRENT_CONTEXT.is_dist_safe or local:
        if enable_cyclic_x:
            arr = update(arr, at[-2:, ...], arr[2:4, ...])
            arr = update(arr, at[:2, ...], arr[-4:-2, ...])
        return arr

    from ..distributed import exchange_overlap
    exchange_overlap(arr, ['xt', 'yt'], cyclic=enable_cyclic_x)
    return arr


@veros_kernel
def pad_z_edges(array):
    """
    Pads the z-axis of an array by repeating its edge values
    """
    if array.ndim == 1:
        newarray = np.pad(array, 1, mode='edge')
    elif array.ndim >= 3:
        newarray = np.pad(array, ((0, 0), (0, 0), (1, 1)), mode='edge')
    else:
        raise ValueError('Array to pad needs to have 1 or at least 3 dimensions')
    return newarray


@veros_kernel(static_args=("nz"))
def create_water_masks(ks, nz):
    ks = ks - 1
    land_mask = ks >= 0
    water_mask = np.logical_and(
        land_mask[:, :, np.newaxis],
        np.arange(nz)[np.newaxis, np.newaxis, :] >= ks[:, :, np.newaxis]
    )
    edge_mask = np.logical_and(
        land_mask[:, :, np.newaxis],
        np.arange(nz)[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis]
    )
    return land_mask, water_mask, edge_mask


@veros_kernel
def solve_implicit(a, b, c, d, water_mask, edge_mask, b_edge=None, d_edge=None):
    if b_edge is not None:
        b = np.where(edge_mask, b_edge, b)

    if d_edge is not None:
        d = np.where(edge_mask, d_edge, d)

    return solve_tridiagonal(a, b, c, d, water_mask, edge_mask)
