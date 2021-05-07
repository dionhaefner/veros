
from veros import logger, veros_kernel, veros_routine, KernelOutput
from veros.variables import allocate
from veros.distributed import global_max
from veros.core import utilities as mainutils
from veros.core.operators import numpy as np, for_loop, update, at
from veros.core.streamfunction import island, line_integrals
from veros.core.streamfunction.solvers import get_linear_solver


@veros_routine
def get_isleperim(state):
    """
    preprocess land map using MOMs algorithm for B-grid to determine number of islands
    """
    from veros.state import resize_dimension
    vs = state.variables

    island.isleperim(state)

    # now that we know the number of islands we can resize
    # all arrays depending on that
    nisle = int(global_max(np.max(vs.land_map)))
    resize_dimension(state, "isle", nisle)


@veros_routine
def streamfunction_init(state):
    """
    prepare for island integrals
    """
    vs = state.variables
    settings = state.settings

    logger.info('Initializing streamfunction method')

    get_isleperim(state)

    boundary_masks_out = boundary_masks(state)
    vs.update(boundary_masks_out)

    # populate linear solver cache
    linear_solver = get_linear_solver(state)

    """
    precalculate time independent boundary components of streamfunction
    """
    forc = allocate(state.dimensions, ("xt", "yt"))

    vs.psin = update(vs.psin, at[...], vs.maskZ[..., -1, np.newaxis])

    for isle in range(state.dimensions["isle"]):
        logger.info(' Solving for boundary contribution by island {:d}'.format(isle))
        isle_sol = linear_solver.solve(state, forc, vs.psin[:, :, isle], boundary_val=vs.boundary_mask[:, :, isle])
        vs.psin = update(vs.psin, at[:, :, isle], isle_sol)

    vs.psin = mainutils.enforce_boundaries(vs.psin, settings.enable_cyclic_x)

    line_psin_out = island_integrals(state)
    vs.update(line_psin_out)


@veros_kernel
def island_integrals(state):
    """
    precalculate time independent island integrals
    """
    vs = state.variables

    uloc = allocate(state.dimensions, ("xt", "yt", "isle"))
    vloc = allocate(state.dimensions, ("xt", "yt", "isle"))

    uloc = update(uloc, at[1:, 1:, :], -(vs.psin[1:, 1:, :] - vs.psin[1:, :-1, :])
        * vs.maskU[1:, 1:, -1, np.newaxis]
        / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.hur[1:, 1:, np.newaxis])
    vloc = update(vloc, at[1:, 1:, ...], (vs.psin[1:, 1:, :] - vs.psin[:-1, 1:, :]) \
        * vs.maskV[1:, 1:, -1, np.newaxis]
        / (vs.cosu[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
        * vs.hvr[1:, 1:, np.newaxis])
    vs.line_psin = line_integrals.line_integrals(
        state, uloc=uloc, vloc=vloc, kind='full'
    )

    return KernelOutput(line_psin=vs.line_psin)


@veros_kernel
def boundary_masks(state):
    """
    now that the number of islands is known we can allocate the rest of the variables
    """
    vs = state.variables
    settings = state.settings

    def loop_body(isle, masks):
        (east_mask, west_mask, south_mask, north_mask, boundary_mask) = masks
        boundary_map = vs.land_map == (isle + 1)

        if settings.enable_cyclic_x:
            east_mask = update(east_mask, at[2:-2, 1:-1, isle], boundary_map[3:-1, 1:-1] & ~boundary_map[3:-1, 2:])
            west_mask = update(west_mask, at[2:-2, 1:-1, isle], boundary_map[2:-2, 2:] & ~boundary_map[2:-2, 1:-1])
            south_mask = update(south_mask, at[2:-2, 1:-1, isle], boundary_map[2:-2, 1:-1] & ~boundary_map[3:-1, 1:-1])
            north_mask = update(north_mask, at[2:-2, 1:-1, isle], boundary_map[3:-1, 2:] & ~boundary_map[2:-2, 2:])
        else:
            east_mask = update(east_mask, at[1:-1, 1:-1, isle], boundary_map[2:, 1:-1] & ~boundary_map[2:, 2:])
            west_mask = update(west_mask, at[1:-1, 1:-1, isle], boundary_map[1:-1, 2:] & ~boundary_map[1:-1, 1:-1])
            south_mask = update(south_mask, at[1:-1, 1:-1, isle], boundary_map[1:-1, 1:-1] & ~boundary_map[2:, 1:-1])
            north_mask = update(north_mask, at[1:-1, 1:-1, isle], boundary_map[2:, 2:] & ~boundary_map[1:-1, 2:])

        boundary_mask = update(boundary_mask, at[..., isle], (
            east_mask[..., isle]
            | west_mask[..., isle]
            | north_mask[..., isle]
            | south_mask[..., isle]
        ))
        return (east_mask, west_mask, south_mask, north_mask, boundary_mask)

    (vs.line_dir_east_mask, vs.line_dir_west_mask, vs.line_dir_south_mask, vs.line_dir_north_mask, vs.boundary_mask) = for_loop(
        0, state.dimensions["isle"], loop_body,
        (vs.line_dir_east_mask, vs.line_dir_west_mask, vs.line_dir_south_mask, vs.line_dir_north_mask, vs.boundary_mask)
    )

    return KernelOutput(
        boundary_mask=vs.boundary_mask, line_dir_east_mask=vs.line_dir_east_mask, line_dir_west_mask=vs.line_dir_west_mask,
        line_dir_south_mask=vs.line_dir_south_mask, line_dir_north_mask=vs.line_dir_north_mask,
    )
