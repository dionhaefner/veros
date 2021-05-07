from veros.core.operators import numpy as np

from veros import veros_kernel
from veros.distributed import global_sum
from veros.core.operators import update, at, for_loop


@veros_kernel(static_args=("kind"))
def line_integrals(state, uloc, vloc, kind='same'):
    """
    calculate line integrals along all islands

    Arguments:
        kind: 'same' calculates only line integral contributions of an island with itself,
               while 'full' calculates all possible pairings between all islands.
    """
    vs = state.variables
    nisle = state.dimensions["isle"]

    east = vloc[1:-2, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 2:-1, np.newaxis]
    west = -vloc[2:-1, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 1:-2, np.newaxis]
    north = vloc[1:-2, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        - uloc[1:-2, 1:-2, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 1:-2, np.newaxis]
    south = -vloc[2:-1, 1:-2, :] * vs.dyu[np.newaxis, 1:-2, np.newaxis] \
        + uloc[1:-2, 2:-1, :] \
        * vs.dxu[1:-2, np.newaxis, np.newaxis] \
        * vs.cost[np.newaxis, 2:-1, np.newaxis]

    if kind == 'same':
        east = np.sum(east * (vs.line_dir_east_mask[1:-2, 1:-2] &
                                vs.boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        west = np.sum(west * (vs.line_dir_west_mask[1:-2, 1:-2] &
                                vs.boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        north = np.sum(north * (vs.line_dir_north_mask[1:-2, 1:-2]
                                & vs.boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        south = np.sum(south * (vs.line_dir_south_mask[1:-2, 1:-2]
                                & vs.boundary_mask[1:-2, 1:-2]), axis=(0, 1))
        return global_sum(east + west + north + south)

    elif kind == 'full':
        isle_int = np.empty((nisle, nisle))

        def loop_body(isle, isle_int):
            east_isle = np.sum(
                east[..., isle, np.newaxis]
                * (vs.line_dir_east_mask[1:-2, 1:-2] & vs.boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            west_isle = np.sum(
                west[..., isle, np.newaxis]
                * (vs.line_dir_west_mask[1:-2, 1:-2] & vs.boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            north_isle = np.sum(
                north[..., isle, np.newaxis]
                * (vs.line_dir_north_mask[1:-2, 1:-2] & vs.boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            south_isle = np.sum(
                south[..., isle, np.newaxis]
                * (vs.line_dir_south_mask[1:-2, 1:-2] & vs.boundary_mask[1:-2, 1:-2]),
                axis=(0, 1)
            )
            isle_int = update(isle_int, at[:, isle], east_isle + west_isle + north_isle + south_isle)
            return isle_int

        isle_int = for_loop(0, nisle, loop_body, isle_int)

        return global_sum(isle_int)

    else:
        raise ValueError('"kind" argument must be "same" or "full"')
