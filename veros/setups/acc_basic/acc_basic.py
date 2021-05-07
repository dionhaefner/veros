#!/usr/bin/env python

from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.distributed import global_min, global_max
from veros.core.operators import numpy as np, update, at


class ACCBasicSetup(VerosSetup):
    """A model using spherical coordinates with a partially closed domain representing the Atlantic and ACC.

    Wind forcing over the channel part and buoyancy relaxation drive a large-scale meridional overturning circulation.

    This setup demonstrates:
     - setting up an idealized geometry
     - updating surface forcings
     - constant horizontal and vertical mixing (switched off IDEMIX and GM_EKE)
     - basic usage of diagnostics

    :doc:`Adapted from ACC channel model </reference/setups/acc>`.
    """
    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        settings.identifier = 'acc_basic'

        settings.nx, settings.ny, settings.nz = 30, 42, 15
        settings.dt_mom = 4800
        settings.dt_tracer = 86400 / 2.
        settings.runlen = 86400 * 365 * 20

        settings.x_origin = 0.0
        settings.y_origin = -40.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.congr_epsilon = 1e-12
        settings.congr_max_iterations = 5000

        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 500.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.01
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = (2 * settings.degtom)**3 * 2e-11
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1

        settings.enable_bottom_friction = True
        settings.r_bot = 1e-5

        settings.enable_implicit_vert_friction = True

        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_Prandtl_tke = False
        settings.enable_kappaH_profile = True

        settings.K_gm_0 = 1000.0
        settings.enable_eke = False
        settings.enable_idemix = False

        settings.eq_of_state_type = 3

        var_meta = state.var_meta
        var_meta.update(
            t_star=Variable("t_star", ("yt",), "deg C", "Reference surface temperature"),
            t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables

        ddz = np.array([50., 70., 100., 140., 190., 240., 290., 340.,
                        390., 440., 490., 540., 590., 640., 690.])
        vs.dxt = update(vs.dxt, at[...], 2.0)
        vs.dyt = update(vs.dyt, at[...], 2.0)
        vs.dzt = ddz[::-1] / 2.5

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(vs.coriolis_t, at[:, :], 2 * settings.omega * np.sin(vs.yt[None, :] / 180. * settings.pi))

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        x, y = np.meshgrid(vs.xt, vs.yt, indexing='ij')
        vs.kbot = np.logical_or(x > 1.0, y < -20).astype(np.int)

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        # initial conditions
        vs.temp = update(vs.temp, at[...], ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None])
        vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

        # wind stress forcing
        yt_min = global_min(vs.yt.min())
        yu_min = global_min(vs.yu.min())
        yt_max = global_max(vs.yt.max())
        yu_max = global_max(vs.yu.max())

        taux = allocate(state.dimensions, ('yt',))
        taux = np.where(vs.yt < -20, 0.1 * np.sin(settings.pi * (vs.yu - yu_min) / (-20.0 - yt_min)), taux)
        taux = np.where(vs.yt > 10, 0.1 * (1 - np.cos(2 * settings.pi * (vs.yu - 10.0) / (yu_max - 10.0))), taux)
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        vs.t_star = allocate(state.dimensions, ('yt',), fill=15)
        vs.t_star = np.where(vs.yt < -20, 15 * (vs.yt - yt_min) / (-20 - yt_min), vs.t_star)
        vs.t_star = np.where(vs.yt > 20, 15 * (1 - (vs.yt - 20) / (yt_max - 20)), vs.t_star)
        vs.t_rest = vs.dzt[np.newaxis, -1] / (30. * 86400.) * vs.maskT[:, :, -1]

        if settings.enable_tke:
            vs.forc_tke_surface = update(vs.forc_tke_surface, at[2:-2, 2:-2], np.sqrt((0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0)**2
                                                      + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0)**2)**(1.5))

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        diagnostics = state.diagnostics

        diagnostics['averages'].output_variables = (
            'salt', 'temp', 'u', 'v', 'w', 'psi', 'surface_taux', 'surface_tauy'
        )
        diagnostics['averages'].output_frequency = 365 * 86400.
        diagnostics['averages'].sampling_frequency = settings.dt_tracer * 10
        diagnostics['overturning'].output_frequency = 365 * 86400. / 48.
        diagnostics['overturning'].sampling_frequency = settings.dt_tracer * 10
        diagnostics['tracer_monitor'].output_frequency = 365 * 86400. / 12.

    @veros_routine
    def after_timestep(self, state):
        pass
