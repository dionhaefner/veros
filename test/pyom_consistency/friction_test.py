import os

import numpy as np

import pytest

from test_base import VerosPyOMUnitTest
from veros.core import friction, numerics


class FrictionTest(VerosPyOMUnitTest):
    nx, ny, nz = 70, 60, 50
    extra_settings = {
        'enable_cyclic_x': True,
        'enable_conserve_energy': True,
        'enable_bottom_friction_var': True,
        'enable_hor_friction_cos_scaling': True,
        'enable_momentum_sources': True,
    }

    def initialize(self):
        self.set_attribute('hor_friction_cosPower', np.random.randint(1, 5))

        for a in ('dt_mom', 'r_bot', 'r_quad_bot', 'A_h', 'A_hbi', 'x_origin', 'y_origin'):
            self.set_attribute(a, np.random.rand())

        for a in ('dxt', 'dxu'):
            self.set_attribute(a, np.ones(self.nx + 4) * np.random.rand())

        for a in ('dyt', 'dyu'):
            self.set_attribute(a, np.ones(self.ny + 4) * np.random.rand())

        for a in ('cosu', 'cost'):
            self.set_attribute(a, np.random.rand(self.ny + 4) + 1)

        for a in ('dzt', 'dzw', 'zw'):
            self.set_attribute(a, 1 + np.random.rand(self.nz))

        for a in ('r_bot_var_u', 'r_bot_var_v'):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4))

        for a in ('area_u', 'area_v', 'area_t'):
            self.set_attribute(a, np.random.rand(self.nx + 4, self.ny + 4))

        for a in ('K_diss_v', 'kappaM', 'flux_north', 'flux_east', 'flux_top', 'K_diss_bot', 'K_diss_h',
                  'du_mix', 'dv_mix', 'u_source', 'v_source'):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz))

        for a in ('u', 'v', 'w'):
            self.set_attribute(a, np.random.randn(self.nx + 4, self.ny + 4, self.nz, 3))

        for a in ('maskU', 'maskV', 'maskW', 'maskT'):
            self.set_attribute(a, np.random.randint(0, 2, size=(self.nx + 4, self.ny + 4, self.nz)).astype(np.float))

        kbot = np.random.randint(1, self.nz, size=(self.nx + 4, self.ny + 4))
        # add some islands, but avoid boundaries
        kbot[3:-3, 3:-3].flat[np.random.randint(0, (self.nx - 2) * (self.ny - 2), size=10)] = 0
        self.set_attribute('kbot', kbot)

        numerics.calc_grid(self.veros_new.state)
        numerics.calc_topo(self.veros_new.state)
        self.veros_legacy.call_fortran_routine('calc_grid')
        self.veros_legacy.call_fortran_routine('calc_topo')

        self.test_module = friction
        veros_args = (self.veros_new.state, )
        veros_legacy_args = dict()
        self.test_routines = {k: (veros_args, veros_legacy_args) for k in (
            'explicit_vert_friction', 'implicit_vert_friction', 'rayleigh_friction',
            'linear_bottom_friction', 'quadratic_bottom_friction', 'harmonic_friction',
            'biharmonic_friction', 'momentum_sources'
        )}

    def test_passed(self, routine):
        for f in ('flux_east', 'flux_north', 'flux_top', 'u', 'v', 'w', 'K_diss_v',
                  'K_diss_bot', 'K_diss_h', 'du_mix', 'dv_mix'):
            self.check_variable(f)


def test_friction(pyom2_lib, backend):
    FrictionTest(fortran=pyom2_lib, backend=backend).run()
