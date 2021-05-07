from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.distributed import global_min, global_max
from veros.core.operators import numpy as np, update, at
import veros.tools


class ACCSectorSetup(VerosSetup):
    """A model using spherical coordinates with a partially closed domain representing the narrow sector of Atlantic and ACC.

    The bathymetry of the model is idealized to a flat-bottom (with depth of 4000 m) over the majority of the domain,
    except a half depth appended within the confines of the circumpolar channel at the inflow and outflow regions.
    The horizontal grid has resolution of :math:`2 \\times 2` degrees, and the vertical one has 40 levels.

    Wind forcing over the sector part and buoyancy relaxation drive a large-scale meridional overturning circulation.

    This setup demonstrates:
     - setting up an idealized geometry after `(Munday et al., 2013) <https://doi.org/10.1175/JPO-D-12-095.1>`_.
     - modifing surface forcings over selected regions of the domain
     - sensitivity of circumpolar transport and meridional overturning
       to changes in Southern Ocean wind stress and buoyancy anomalies
     - basic usage of diagnostics

    :doc:`Adapted from ACC channel model </reference/setups/acc>`.

    Reference:

        Laurits S. Andreasen. (2019). Time scales of the Bipolar seesaw: The role of oceanic cross-hemisphere signals,
            Southern Ocean eddies and wind changes, MSc Thesis, 42p.
            `<https://sid.erda.dk/share_redirect/CVvcrowL22/Thesis/Laurits_Andreasen_MSc_thesis.pdf>`_.

    """
    max_depth = 4000.

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings

        settings.identifier = 'acc_sector'

        settings.nx, settings.ny, settings.nz = 15, 62, 40
        settings.dt_mom = 3600.
        settings.dt_tracer = 3600.
        settings.runlen = 86400 * 365

        settings.x_origin = 0.0
        settings.y_origin = -60.0

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
        settings.A_h = 5e4 * 2
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

        settings.K_gm_0 = 1300.0
        settings.enable_eke = False
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = False
        settings.enable_eke_isopycnal_diffusion = False

        settings.enable_idemix = False
        settings.enable_idemix_hor_diffusion = False
        settings.enable_eke_diss_surfbot = False
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = False

        settings.eq_of_state_type = 3

        var_meta = state.var_meta
        var_meta.update(
            t_star=Variable("t_star", ("yt",), "deg C", "Reference surface temperature"),
            t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        # keep total domain size constant when nx or ny changes
        vs.dxt = update(vs.dxt, at[...], 2.0 * 15 / settings.nx)
        vs.dyt = update(vs.dyt, at[...], 2.0 * 62 / settings.ny)
        vs.dzt = veros.tools.get_vinokur_grid_steps(settings.nz, self.max_depth, 10., refine_towards='lower')

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(vs.coriolis_t, at[:, :], 2 * settings.omega * np.sin(vs.yt[None, :] / 180. * settings.pi))

    @veros_routine
    def set_topography(self, state):
        vs = state.variables

        x, y = np.meshgrid(vs.xt, vs.yt, indexing='ij')
        vs.kbot = np.logical_or((x > 1.0) & (x < 27), y < -40).astype('int')

        # A half depth (ridge) is appended to the domain within the confines
        # of the circumpolar channel at the inflow and outflow regions
        bathymetry = np.logical_or(((x <= 1.0) & (y < -40)), ((x >= 27) & (y < -40)))
        kzt2000 = np.sum((vs.zt < -2000.).astype('int'))
        vs.kbot = np.where(bathymetry, kzt2000, vs.kbot)

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
        north = vs.yt > 30
        subequatorial_north_n = (vs.yt >= 15) & (vs.yt < 30)
        subequatorial_north_s = (vs.yt > 0) & (vs.yt < 15)
        equator = (vs.yt > -5) & (vs.yt < 5)
        subequatorial_south_n = (vs.yt > -15) & (vs.yt < 0)
        subequatorial_south_s = (vs.yt <= -15) & (vs.yt > -30)
        south = vs.yt < -30

        taux = np.where(north, -5e-2 * np.sin(settings.pi * (vs.yu - yu_max) / (yt_max - 30.)), taux)
        taux = np.where(subequatorial_north_s, 5e-2 * np.sin(settings.pi * (vs.yu - 30.) / 30.), taux)
        taux = np.where(subequatorial_north_n, 5e-2 * np.sin(settings.pi * (vs.yt - 30.) / 30.), taux)
        taux = np.where(subequatorial_south_n, -5e-2 * np.sin(settings.pi * (vs.yu - 30.) / 30.), taux)
        taux = np.where(subequatorial_south_s, -5e-2 * np.sin(settings.pi * (vs.yt - 30.) / 30.), taux)
        taux = np.where(equator, -1.5e-2 * np.cos(settings.pi * (vs.yu - 10.) / 10.) - 2.5e-2, taux)
        taux = np.where(south, 15e-2 * np.sin(settings.pi * (vs.yu - yu_min) / (-30. - yt_min)), taux)
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        delta_t, ts, tn = 25., 0., 5.
        vs.t_star = allocate(state.dimensions, ('yt',), fill=delta_t)
        vs.t_star = np.where(vs.yt < 0, ts + delta_t * np.sin(settings.pi * (vs.yt + 60.) / np.abs(2 * settings.y_origin)), vs.t_star)
        vs.t_star = np.where(vs.yt > 0, tn + (delta_t + ts - tn) * np.sin(settings.pi * (vs.yt + 60.) / np.abs(2 * settings.y_origin)), vs.t_star)
        vs.t_rest = vs.dzt[-1] / (10. * 86400.) * vs.maskT[:, :, -1]

        if settings.enable_tke:
            vs.forc_tke_surface = update(vs.forc_tke_surface, at[2:-2, 2:-2], np.sqrt((0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0)**2
                                                      + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0)**2)**(1.5))

        if settings.enable_idemix:
            vs.forc_iw_bottom = 1e-6 * vs.maskW[:, :, -1]
            vs.forc_iw_surface = 1e-7 * vs.maskW[:, :, -1]

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        state.diagnostics['snapshot'].output_frequency = 86400 * 10
        state.diagnostics['averages'].output_variables = (
            'salt', 'temp', 'u', 'v', 'w', 'psi', 'rho', 'surface_taux', 'surface_tauy'
        )
        state.diagnostics['averages'].output_frequency = 365 * 86400.
        state.diagnostics['averages'].sampling_frequency = settings.dt_tracer * 10
        state.diagnostics['overturning'].output_frequency = 365 * 86400. / 48.
        state.diagnostics['overturning'].sampling_frequency = settings.dt_tracer * 10
        state.diagnostics['tracer_monitor'].output_frequency = 365 * 86400. / 12.
        state.diagnostics['energy'].output_frequency = 365 * 86400. / 48
        state.diagnostics['energy'].sampling_frequency = settings.dt_tracer * 10

    @veros_routine
    def after_timestep(self, state):
        pass
