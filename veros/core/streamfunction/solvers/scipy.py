import numpy as onp
import scipy.sparse
import scipy.sparse.linalg as spalg

from veros import logger
from veros import distributed, veros_routine, runtime_state as rst
from veros.variables import allocate
from veros.core import utilities
from veros.core.operators import update, at, numpy as np
from veros.core.streamfunction.solvers.base import LinearSolver


class SciPySolver(LinearSolver):
    @veros_routine(
        local_variables=(
            'hvr', 'hur',
            'dxu', 'dxt', 'dyu', 'dyt',
            'cosu', 'cost',
            'boundary_mask'
        ),
        dist_safe=False,
    )
    def __init__(self, state):
        self._matrix = self._assemble_poisson_matrix(state)
        jacobi_precon = self._jacobi_preconditioner(state, self._matrix)
        self._matrix = jacobi_precon * self._matrix
        self._rhs_scale = jacobi_precon.diagonal()
        self._extra_args = {}

        logger.info('Computing ILU preconditioner...')
        ilu_preconditioner = spalg.spilu(self._matrix.tocsc(), drop_tol=1e-6, fill_factor=100)
        self._extra_args['M'] = spalg.LinearOperator(self._matrix.shape, ilu_preconditioner.solve)

    def _scipy_solver(self, state, rhs, x0, boundary_mask, boundary_val):
        settings = state.settings

        orig_shape = x0.shape
        x0 = utilities.enforce_boundaries(x0, settings.enable_cyclic_x, local=True)

        boundary_mask = ~np.any(boundary_mask, axis=2)
        rhs = np.where(boundary_mask, rhs, boundary_val) # set right hand side on boundaries

        rhs = onp.asarray(rhs.reshape(-1), dtype='float64') * self._rhs_scale
        x0 = onp.asarray(x0.reshape(-1), dtype='float64')

        linear_solution, info = spalg.bicgstab(
            self._matrix, rhs,
            x0=x0, atol=0, tol=settings.congr_epsilon,
            maxiter=settings.congr_max_iterations,
            **self._extra_args
        )

        if info > 0:
            logger.warning('Streamfunction solver did not converge after {} iterations', info)

        return np.asarray(linear_solution.reshape(orig_shape))

    def solve(self, state, rhs, x0, boundary_val=None):
        """
        Main solver for streamfunction. Solves a 2D Poisson equation. Uses scipy.sparse.linalg
        linear solvers.

        Arguments:
            rhs: Right-hand side vector
            x0: Initial guess
            boundary_val: Array containing values to set on boundary elements. Defaults to `x0`.

        """
        vs = state.variables

        rhs_global = distributed.gather(rhs, state.dimensions, ('xt', 'yt'))
        x0_global = distributed.gather(x0, state.dimensions, ('xt', 'yt'))
        boundary_mask_global = distributed.gather(vs.boundary_mask, state.dimensions, ('xt', 'yt'))

        if boundary_val is None:
            boundary_val = x0_global
        else:
            boundary_val = distributed.gather(boundary_val, state.dimensions, ('xt', 'yt'))

        if rst.proc_rank == 0:
            linear_solution = self._scipy_solver(
                state, rhs_global, x0_global, boundary_mask=boundary_mask_global, boundary_val=boundary_val
            )
        else:
            linear_solution = np.empty_like(rhs)

        return distributed.scatter(linear_solution, state.dimensions, ('xt', 'yt'))

    @staticmethod
    def _jacobi_preconditioner(state, matrix):
        """
        Construct a simple Jacobi preconditioner
        """
        settings = state.settings

        eps = 1e-20
        Z = allocate(state.dimensions, ('xu', 'yu'), fill=1, local=False)
        Y = np.reshape(matrix.diagonal().copy(), (settings.nx + 4, settings.ny + 4))[2:-2, 2:-2]
        Z = update(Z, at[2:-2, 2:-2], np.where(np.abs(Y) > eps, 1. / (Y + eps), 1.))
        return scipy.sparse.dia_matrix((Z.reshape(-1), 0), shape=(Z.size, Z.size)).tocsr()

    @staticmethod
    def _assemble_poisson_matrix(state):
        """
        Construct a sparse matrix based on the stencil for the 2D Poisson equation.
        """
        vs = state.variables
        settings = state.settings

        boundary_mask = ~np.any(vs.boundary_mask, axis=2)

        # assemble diagonals
        main_diag = allocate(state.dimensions, ('xu', 'yu'), fill=1, local=False)
        east_diag, west_diag, north_diag, south_diag = (allocate(state.dimensions, ('xu', 'yu'), local=False) for _ in range(4))
        main_diag = update(main_diag, at[2:-2, 2:-2], -vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, np.newaxis] / vs.dxt[3:-1, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2 \
            - vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, np.newaxis] / vs.dxt[2:-2, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2 \
            - vs.hur[2:-2, 2:-2] / vs.dyu[np.newaxis, 2:-2] / vs.dyt[np.newaxis, 2:-2] * vs.cost[np.newaxis, 2:-2] / vs.cosu[np.newaxis, 2:-2] \
            - vs.hur[2:-2, 3:-1] / vs.dyu[np.newaxis, 2:-2] / vs.dyt[np.newaxis, 3:-1] * vs.cost[np.newaxis, 3:-1] / vs.cosu[np.newaxis, 2:-2])
        east_diag = update(east_diag, at[2:-2, 2:-2], vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, np.newaxis] / \
            vs.dxt[3:-1, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2)
        west_diag = update(west_diag, at[2:-2, 2:-2], vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, np.newaxis] / \
            vs.dxt[2:-2, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2)
        north_diag = update(north_diag, at[2:-2, 2:-2], vs.hur[2:-2, 3:-1] / vs.dyu[np.newaxis, 2:-2] / \
            vs.dyt[np.newaxis, 3:-1] * vs.cost[np.newaxis, 3:-1] / vs.cosu[np.newaxis, 2:-2])
        south_diag = update(south_diag, at[2:-2, 2:-2], vs.hur[2:-2, 2:-2] / vs.dyu[np.newaxis, 2:-2] / \
            vs.dyt[np.newaxis, 2:-2] * vs.cost[np.newaxis, 2:-2] / vs.cosu[np.newaxis, 2:-2])

        if settings.enable_cyclic_x:
            # couple edges of the domain
            wrap_diag_east, wrap_diag_west = (allocate(state.dimensions, ('xu', 'yu'), local=False) for _ in range(2))
            wrap_diag_east = update(wrap_diag_east, at[2, 2:-2], west_diag[2, 2:-2] * boundary_mask[2, 2:-2])
            wrap_diag_west = update(wrap_diag_west, at[-3, 2:-2], east_diag[-3, 2:-2] * boundary_mask[-3, 2:-2])
            west_diag = update(west_diag, at[2, 2:-2], 0.)
            east_diag = update(east_diag, at[-3, 2:-2], 0.)

        main_diag = main_diag * boundary_mask
        main_diag = np.where(main_diag == 0., 1., main_diag)

        # construct sparse matrix
        cf = tuple(diag.reshape(-1) for diag in (
            main_diag,
            boundary_mask * east_diag,
            boundary_mask * west_diag,
            boundary_mask * north_diag,
            boundary_mask * south_diag
        ))
        offsets = (0, -main_diag.shape[1], main_diag.shape[1], -1, 1)

        if settings.enable_cyclic_x:
            offsets += (-main_diag.shape[1] * (settings.nx - 1), main_diag.shape[1] * (settings.nx - 1))
            cf += (wrap_diag_east.reshape(-1), wrap_diag_west.reshape(-1))

        cf = onp.asarray(cf, dtype='float64')
        return scipy.sparse.dia_matrix((cf, offsets), shape=(main_diag.size, main_diag.size)).T.tocsr()
