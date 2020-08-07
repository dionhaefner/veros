from loguru import logger
import numpy as np

from veros import (
    veros_kernel, veros_routine, run_kernel,
    runtime_settings as rs, runtime_state as rst
)
from veros.core import utilities as mainutils
from veros.core.streamfunction import island, utilities


@veros_routine(
    inputs=('kbot', 'land_map'),
    outputs=('land_map'),
    settings=('enable_cyclic_x'),
    dist_safe=False
)
def get_isleperim(vs):
    """
    preprocess land map using MOMs algorithm for B-grid to determine number of islands
    """
    logger.debug(' Determining number of land masses')
    land_map = island.isleperim(vs.kbot, vs.enable_cyclic_x)
    logger.info(_ascii_map(land_map))
    return dict(land_map=land_map)


def _get_solver_class():
    ls = rs.linear_solver

    def _get_best_solver():
        if rst.proc_num > 1:
            try:
                from .solvers.petsc import PETScSolver
            except ImportError:
                logger.warning('PETSc linear solver not available, falling back to SciPy')
            else:
                return PETScSolver

        from .solvers.scipy import SciPySolver
        return SciPySolver

    if ls == 'best':
        return _get_best_solver()
    elif ls == 'petsc':
        from .solvers.petsc import PETScSolver
        return PETScSolver
    elif ls == 'scipy':
        from .solvers.scipy import SciPySolver
        return SciPySolver

    raise ValueError('unrecognized linear solver %s' % ls)


@veros_routine(
    inputs=(
        'land_map', 'nisle', 'kbot', 'nx', 'ny', 'isle', 'psin', 'dpsin', 'line_psin', 'boundary_mask',
        'line_dir_south_mask', 'line_dir_north_mask', 'line_dir_east_mask',
        'line_dir_west_mask', 'linear_solver', 'dxt', 'dyt', 'hur', 'hvr', 'maskZ',
        'maskU', 'maskV', 'cosu', 'dxu', 'dyu', 'cost'
    ),
    outputs=(
        'psin', 'dpsin', 'line_psin', 'boundary_mask', 'line_dir_south_mask',
        'line_dir_north_mask', 'line_dir_east_mask', 'line_dir_west_mask', 'linear_solver',
        'nisle', 'isle'
    ),
    settings=(
        'default_float_type', 'enable_cyclic_x'
    ),
    subroutines=(get_isleperim)
)
def streamfunction_init(vs):
    """
    prepare for island integrals
    """
    logger.info('Initializing streamfunction method')

    get_isleperim(vs)
    nisle = np.max(vs.land_map)

    vs.linear_solver = _get_solver_class()(vs)

    (boundary_mask, line_dir_east_mask, line_dir_west_mask, line_dir_south_mask,
     line_dir_north_mask) = run_kernel(boundary_masks, vs, nisle=nisle)

    """
    precalculate time independent boundary components of streamfunction
    """
    forc = np.zeros((vs.nx + 4, vs.ny + 4), dtype=vs.default_float_type)
    psin = np.zeros((vs.nx + 4, vs.ny + 4, nisle), dtype=vs.default_float_type)
    dpsin = np.zeros((nisle, 3), dtype=vs.default_float_type)

    psin[...] = vs.maskZ[..., -1, np.newaxis]

    for isle in range(nisle):
        logger.info(' Solving for boundary contribution by island {:d}'.format(isle))
        vs.linear_solver.solve(vs, forc, psin[:, :, isle], boundary_val=boundary_mask[:, :, isle])

    mainutils.enforce_boundaries(psin, vs.enable_cyclic_x)

    line_psin = run_kernel(
        island_integrals, vs, nisle=nisle, psin=psin,
        boundary_mask=boundary_mask, line_dir_east_mask=line_dir_east_mask, line_dir_west_mask=line_dir_west_mask,
        line_dir_south_mask=line_dir_south_mask, line_dir_north_mask=line_dir_north_mask
    )

    isle = np.arange(1, nisle + 1)

    return dict(
        psin=psin, dpsin=dpsin, line_psin=line_psin, boundary_mask=boundary_mask,
        line_dir_south_mask=line_dir_south_mask, line_dir_north_mask=line_dir_north_mask,
        line_dir_east_mask=line_dir_east_mask, line_dir_west_mask=line_dir_west_mask,
        linear_solver=vs.linear_solver,
        nisle=nisle, isle=isle
    )


@veros_kernel(static_args=('default_float_type'))
def island_integrals(nx, ny, nisle, default_float_type, maskU, maskV, dxt, dyt, dxu, dyu, cost, cosu, hur, hvr, psin,
                     line_dir_east_mask, line_dir_west_mask, line_dir_north_mask, line_dir_south_mask, boundary_mask):
    """
    precalculate time independent island integrals
    """
    fpx = np.zeros((nx + 4, ny + 4, nisle), dtype=default_float_type)
    fpy = np.zeros((nx + 4, ny + 4, nisle), dtype=default_float_type)

    fpx[1:, 1:, :] = -maskU[1:, 1:, -1, np.newaxis] \
        * (psin[1:, 1:, :] - psin[1:, :-1, :]) \
        / dyt[np.newaxis, 1:, np.newaxis] * hur[1:, 1:, np.newaxis]
    fpy[1:, 1:, ...] = maskV[1:, 1:, -1, np.newaxis] \
        * (psin[1:, 1:, :] - psin[:-1, 1:, :]) \
        / (cosu[np.newaxis, 1:, np.newaxis] * dxt[1:, np.newaxis, np.newaxis]) \
        * hvr[1:, 1:, np.newaxis]
    line_psin = utilities.line_integrals(
        dxu, dyu, cost, line_dir_east_mask,
        line_dir_west_mask, line_dir_north_mask, line_dir_south_mask, boundary_mask,
        nisle, uloc=fpx, vloc=fpy, kind='full'
    )

    return line_psin


@veros_kernel(static_args=('enable_cyclic_x'))
def boundary_masks(land_map, nx, ny, nisle, enable_cyclic_x):
    """
    now that the number of islands is known we can allocate the rest of the variables
    """
    boundary_mask = np.zeros((nx + 4, ny + 4, nisle), dtype='bool')
    line_dir_south_mask = np.zeros((nx + 4, ny + 4, nisle), dtype='bool')
    line_dir_north_mask = np.zeros((nx + 4, ny + 4, nisle), dtype='bool')
    line_dir_east_mask = np.zeros((nx + 4, ny + 4, nisle), dtype='bool')
    line_dir_west_mask = np.zeros((nx + 4, ny + 4, nisle), dtype='bool')

    for isle in range(nisle):
        boundary_map = land_map == (isle + 1)

        if enable_cyclic_x:
            line_dir_east_mask[2:-2, 1:-1, isle] = boundary_map[3:-1, 1:-1] & ~boundary_map[3:-1, 2:]
            line_dir_west_mask[2:-2, 1:-1, isle] = boundary_map[2:-2, 2:] & ~boundary_map[2:-2, 1:-1]
            line_dir_south_mask[2:-2, 1:-1, isle] = boundary_map[2:-2, 1:-1] & ~boundary_map[3:-1, 1:-1]
            line_dir_north_mask[2:-2, 1:-1, isle] = boundary_map[3:-1, 2:] & ~boundary_map[2:-2, 2:]
        else:
            line_dir_east_mask[1:-1, 1:-1, isle] = boundary_map[2:, 1:-1] & ~boundary_map[2:, 2:]
            line_dir_west_mask[1:-1, 1:-1, isle] = boundary_map[1:-1, 2:] & ~boundary_map[1:-1, 1:-1]
            line_dir_south_mask[1:-1, 1:-1, isle] = boundary_map[1:-1, 1:-1] & ~boundary_map[2:, 1:-1]
            line_dir_north_mask[1:-1, 1:-1, isle] = boundary_map[2:, 2:] & ~boundary_map[1:-1, 2:]

        boundary_mask[..., isle] = (
            line_dir_east_mask[..., isle]
            | line_dir_west_mask[..., isle]
            | line_dir_north_mask[..., isle]
            | line_dir_south_mask[..., isle]
        )

    return boundary_mask, line_dir_east_mask, line_dir_west_mask, line_dir_south_mask, line_dir_north_mask


def _ascii_map(boundary_map):
    def _get_char(c):
        if c == 0:
            return '.'
        if c < 0:
            return '#'
        return str(c % 10)

    nx, ny = boundary_map.shape
    map_string = ''
    linewidth = 100
    iremain = nx
    istart = 0
    map_string += '\n'
    map_string += ' ' * (5 + min(linewidth, nx) // 2 - 13) + 'Land mass and perimeter'
    map_string += '\n'
    for isweep in range(1, nx // linewidth + 2):
        iline = min(iremain, linewidth)
        iremain = iremain - iline
        if iline > 0:
            map_string += '\n'
            map_string += ''.join(['{:5d}'.format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += '\n'
            for j in range(ny - 1, -1, -1):
                map_string += '{:3d} '.format(j)
                map_string += ''.join([_get_char(boundary_map[istart + i - 2, j]) for i in range(2, iline + 2)])
                map_string += '\n'
            map_string += ''.join(['{:5d}'.format(istart + i + 1 - 2) for i in range(1, iline + 1, 5)])
            map_string += '\n'
            istart = istart + iline
    map_string += '\n'
    return map_string
