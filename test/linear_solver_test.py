import pytest

import numpy as np

from veros import VerosState
from veros.core.streamfunction.solvers import (
    scipy as scipysolver,
    petsc as petscsolver
)


class SolverTestState(VerosState):
    def __init__(self, cyclic):
        self.nx = 400
        self.ny = 200
        self.nz = 1
        self.nisle = 0

        self.default_float_type = 'float64'

        self.congr_epsilon = 1e-12
        self.congr_max_iterations = 10000

        self.enable_cyclic_x = cyclic

        self.dxt = 1e-12 * np.ones(self.nx + 4)
        self.dxu = 1e-12 * np.ones(self.nx + 4)

        self.dyt = 1e-12 * np.ones(self.ny + 4)
        self.dyu = 1e-12 * np.ones(self.ny + 4)

        self.hur = np.linspace(500, 2000, self.nx + 4)[:, None] * np.ones((self.nx + 4, self.ny + 4))
        self.hvr = np.linspace(500, 2000, self.ny + 4)[None, :] * np.ones((self.nx + 4, self.ny + 4))

        self.cosu = np.ones(self.ny + 4)
        self.cost = np.ones(self.ny + 4)

        self.boundary_mask = np.zeros((self.nx + 4, self.ny + 4, self.nz), dtype='bool')
        self.boundary_mask[:, :2] = 1
        self.boundary_mask[50:100, 50:100] = 1


def get_residual(vs, rhs, sol, boundary_val=None):
    scipy_solver = scipysolver.SciPySolver(vs)
    if boundary_val is None:
        boundary_val = sol
    boundary_mask = np.logical_and.reduce(~vs.boundary_mask, axis=2)
    rhs = np.where(boundary_mask, rhs, boundary_val)
    residual = scipy_solver._matrix @ sol.reshape(-1) - rhs.reshape(-1) * scipy_solver._rhs_scale
    return residual


@pytest.mark.parametrize('cyclic', [True, False])
@pytest.mark.parametrize('solver_class', [scipysolver.SciPySolver, petscsolver.PETScSolver])
def test_solver(solver_class, cyclic, backend):
    from veros import runtime_settings as rs
    rs.backend = backend

    vs = SolverTestState(cyclic)

    rhs = np.ones((vs.nx + 4, vs.ny + 4))
    sol = np.random.rand(vs.nx + 4, vs.ny + 4)

    solver_class(vs).solve(vs, rhs, sol, 10)

    residual = get_residual(vs, rhs, sol, 10)

    # set tolerance may apply in preconditioned space,
    # so let's allow for some wiggle room
    assert np.max(np.abs(residual)) < vs.congr_epsilon * 1e2
