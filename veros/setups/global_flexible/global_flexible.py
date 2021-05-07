#!/usr/bin/env python

import os
import h5netcdf
import scipy.ndimage

from veros import (
    veros_routine, veros_kernel,
    KernelOutput, VerosSetup,
    runtime_settings as rs, runtime_state as rst
)
from veros.variables import Variable, allocate
from veros.core.utilities import enforce_boundaries
from veros.core.operators import numpy as np, update, at
import veros.tools
import veros.time

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets('global_flexible', os.path.join(BASE_PATH, 'assets.json'))


@veros_kernel
def set_forcing_kernel(state):
    vs = state.variables
    settings = state.settings

    t_rest = 30. * 86400.
    cp_0 = 3991.86795711963  # J/kg /K

    year_in_seconds = veros.time.convert_time(1., 'years', 'seconds')
    (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(vs.time, year_in_seconds,
                                                     year_in_seconds / 12., 12)

    # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
    vs.surface_taux = update(vs.surface_taux, at[:-1, :], f1 * vs.taux[1:, :, n1] + f2 * vs.taux[1:, :, n2])
    vs.surface_tauy = update(vs.surface_tauy, at[:, :-1], f1 * vs.tauy[:, 1:, n1] + f2 * vs.tauy[:, 1:, n2])

    if settings.enable_tke:
        vs.forc_tke_surface = update(vs.forc_tke_surface, at[1:-1, 1:-1], np.sqrt((0.5 * (vs.surface_taux[1:-1, 1:-1]
                                                                                            + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                                                                                    + (0.5 * (vs.surface_tauy[1:-1, 1:-1]
                                                                                            + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2) ** (3. / 2.))

    # W/m^2 K kg/J m^3/kg = K m/s
    t_star_cur = f1 * vs.t_star[..., n1] + f2 * vs.t_star[..., n2]
    qqnec = f1 * vs.qnec[..., n1] + f2 * vs.qnec[..., n2]
    qqnet = f1 * vs.qnet[..., n1] + f2 * vs.qnet[..., n2]
    vs.forc_temp_surface = (qqnet + qqnec * (t_star_cur - vs.temp[..., -1, vs.tau])) * vs.maskT[..., -1] / cp_0 / settings.rho_0
    s_star_cur = f1 * vs.s_star[..., n1] + f2 * vs.s_star[..., n2]
    vs.forc_salt_surface = 1. / t_rest * (s_star_cur - vs.salt[..., -1, vs.tau]) * vs.maskT[..., -1] * vs.dzt[-1]

    # apply simple ice mask
    mask1 = vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] > -1.8
    mask2 = vs.forc_temp_surface > 0
    ice = np.logical_or(mask1, mask2)
    vs.forc_temp_surface *= ice
    vs.forc_salt_surface *= ice

    # solar radiation
    if settings.enable_tempsalt_sources:
        vs.temp_source = ((f1 * vs.qsol[..., n1, None] + f2 * vs.qsol[..., n2, None])
                                * vs.divpen_shortwave[None, None, :] * ice[..., None]
                                * vs.maskT[..., :] / cp_0 / settings.rho_0)

    return KernelOutput(
        surface_taux=vs.surface_taux, surface_tauy=vs.surface_tauy, temp_source=vs.temp_source,
        forc_tke_surface=vs.forc_tke_surface, forc_temp_surface=vs.forc_temp_surface, forc_salt_surface=vs.forc_salt_surface,
    )


class GlobalFlexibleResolutionSetup(VerosSetup):
    """
    Global model with flexible resolution.
    """
    # global settings
    min_depth = 10.
    max_depth = 5400.
    equatorial_grid_spacing_factor = 0.5
    polar_grid_spacing_factor = None

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings

        settings.identifier = 'UNNAMED'

        settings.nx = 360
        settings.ny = 160
        settings.nz = 60
        settings.dt_mom = settings.dt_tracer = 900
        settings.runlen = 86400 * 10

        settings.x_origin = 90.
        settings.y_origin = -80.

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        # streamfunction
        settings.congr_epsilon = 1e-10
        settings.congr_max_iterations = 1000

        # friction
        settings.enable_hor_friction = True
        settings.A_h = 5e4
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1
        settings.enable_tempsalt_sources = True
        settings.enable_implicit_vert_friction = True

        settings.eq_of_state_type = 5

        # isoneutral
        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 50.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.005
        settings.enable_skew_diffusion = True

        # tke
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        # eke
        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        # idemix
        settings.enable_idemix = False
        settings.enable_eke_diss_surfbot = True
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = True
        settings.enable_idemix_hor_diffusion = True

        # custom variables
        nmonths = 12
        state.var_meta.update(
            t_star=Variable('t_star', ('xt', 'yt', nmonths), '', '', time_dependent=False),
            s_star=Variable('s_star', ('xt', 'yt', nmonths), '', '', time_dependent=False),
            qnec=Variable('qnec', ('xt', 'yt', nmonths), '', '', time_dependent=False),
            qnet=Variable('qnet', ('xt', 'yt', nmonths), '', '', time_dependent=False),
            qsol=Variable('qsol', ('xt', 'yt', nmonths), '', '', time_dependent=False),
            divpen_shortwave=Variable('divpen_shortwave', ('zt',), '', '', time_dependent=False),
            taux=Variable('taux', ('xt', 'yt', nmonths), '', '', time_dependent=False),
            tauy=Variable('tauy', ('xt', 'yt', nmonths), '', '', time_dependent=False),
        )

    def _get_data(self, vs, var, idx=None):
        if idx is None:
            idx = Ellipsis
        else:
            idx = idx[::-1]

        kwargs = {}
        if rst.proc_num > 1:
            kwargs.update(
                driver='mpio',
                comm=rs.mpi_comm,
            )

        with h5netcdf.File(DATA_FILES['forcing'], 'r', **kwargs) as forcing_file:
            var_obj = forcing_file.variables[var]
            return np.array(var_obj[idx]).T

    @veros_routine(dist_safe=False, local_variables=[
        'dxt', 'dyt', 'dzt'
    ])
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        if settings.ny % 2:
            raise ValueError('ny has to be an even number of grid cells')

        vs.dxt = update(vs.dxt, at[...], 360. / settings.nx)

        if self.equatorial_grid_spacing_factor is not None:
            eq_spacing = self.equatorial_grid_spacing_factor * 160. / settings.ny
        else:
            eq_spacing = None

        if self.polar_grid_spacing_factor is not None:
            polar_spacing = self.polar_grid_spacing_factor * 160. / settings.ny
        else:
            polar_spacing = None

        vs.dyt = update(vs.dyt, at[2:-2], veros.tools.get_vinokur_grid_steps(
            settings.ny, 160., eq_spacing, upper_stepsize=polar_spacing, two_sided_grid=True
        ))
        vs.dzt = veros.tools.get_vinokur_grid_steps(settings.nz, self.max_depth, self.min_depth, refine_towards='lower')

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(vs.coriolis_t, at[...], 2 * settings.omega * np.sin(vs.yt[np.newaxis, :] / 180. * settings.pi))

    def _shift_longitude_array(self, vs, lon, arr):
        wrap_i = np.where((lon[:-1] < vs.xt.min()) & (lon[1:] >= vs.xt.min()))[0][0]
        new_lon = np.concatenate((lon[wrap_i:-1], lon[:wrap_i] + 360.))
        new_arr = np.concatenate((arr[wrap_i:-1, ...], arr[:wrap_i, ...]))
        return new_lon, new_arr

    @veros_routine(dist_safe=False, local_variables=[
        'kbot', 'xt', 'yt', 'zt'
    ])
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings

        with h5netcdf.File(DATA_FILES['topography'], 'r') as topography_file:
            topo_x, topo_y, topo_z = (
                np.array(topography_file.variables[k], dtype='float').T
                for k in ('x', 'y', 'z')
            )

        topo_z = np.minimum(topo_z, 0.)

        # smooth topography to match grid resolution
        gaussian_sigma = (0.5 * len(topo_x) / settings.nx, 0.5 * len(topo_y) / settings.ny)
        topo_z_smoothed = scipy.ndimage.gaussian_filter(topo_z, sigma=gaussian_sigma)
        topo_z_smoothed = np.where(topo_z >= -1, 0, topo_z_smoothed)

        topo_x_shifted, topo_z_shifted = self._shift_longitude_array(vs, topo_x, topo_z_smoothed)
        coords = (vs.xt[2:-2], vs.yt[2:-2])
        z_interp = allocate(state.dimensions, ('xt', 'yt'), local=False)
        z_interp = update(z_interp, at[2:-2, 2:-2], veros.tools.interpolate(
            (topo_x_shifted, topo_y), topo_z_shifted, coords, kind='nearest', fill=False
        ))

        depth_levels = 1 + np.argmin(np.abs(z_interp[:, :, np.newaxis] - vs.zt[np.newaxis, np.newaxis, :]), axis=2)
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], np.where(z_interp < 0., depth_levels, 0)[2:-2, 2:-2])
        vs.kbot = np.where(vs.kbot < settings.nz, vs.kbot, 0)
        vs.kbot = enforce_boundaries(vs.kbot, settings.enable_cyclic_x, local=True)

        # remove marginal seas
        # (dilate to close 1-cell passages, fill holes, undo dilation)
        marginal = (
            scipy.ndimage.binary_erosion(
                scipy.ndimage.binary_fill_holes(
                    scipy.ndimage.binary_dilation(vs.kbot == 0)
                )
            )
        )

        vs.kbot = np.where(marginal, 0, vs.kbot)

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)
        xt_forc, yt_forc, zt_forc = (self._get_data(vs, k) for k in ('xt', 'yt', 'zt'))
        zt_forc = zt_forc[::-1]

        # coordinates must be monotonous for this to work
        assert np.diff(xt_forc).all() > 0
        assert np.diff(yt_forc).all() > 0

        # determine slice to read from forcing file
        data_subset = (
            slice(
                max(0, int(np.argmax(xt_forc >= vs.xt.min())) - 1),
                len(xt_forc) - max(0, int(np.argmax(xt_forc[::-1] <= vs.xt.max())) - 1)
            ),
            slice(
                max(0, int(np.argmax(yt_forc >= vs.yt.min())) - 1),
                len(yt_forc) - max(0, int(np.argmax(yt_forc[::-1] <= vs.yt.max())) - 1)
            ),
            Ellipsis
        )

        xt_forc = xt_forc[data_subset[0]]
        yt_forc = yt_forc[data_subset[1]]

        # initial conditions
        temp_raw = self._get_data(vs, 'temperature', idx=data_subset)[..., ::-1]
        temp_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), temp_raw,
                                            t_grid)
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, :], (temp_data * vs.maskT[2:-2, 2:-2, :])[..., np.newaxis])

        salt_raw = self._get_data(vs, 'salinity', idx=data_subset)[..., ::-1]
        salt_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), salt_raw,
                                            t_grid)
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, :], (salt_data * vs.maskT[2:-2, 2:-2, :])[..., np.newaxis])

        # wind stress on MIT grid
        time_grid = (vs.xt[2:-2], vs.yt[2:-2], np.arange(12))
        taux_raw = self._get_data(vs, 'tau_x', idx=data_subset)
        taux_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            taux_raw, time_grid)
        vs.taux = update(vs.taux, at[2:-2, 2:-2, :], taux_data)

        tauy_raw = self._get_data(vs, 'tau_y', idx=data_subset)
        tauy_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            tauy_raw, time_grid)
        vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], tauy_data)

        vs.taux = enforce_boundaries(vs.taux, settings.enable_cyclic_x)
        vs.tauy = enforce_boundaries(vs.tauy, settings.enable_cyclic_x)

        # Qnet and dQ/dT and Qsol
        qnet_raw = self._get_data(vs, 'q_net', idx=data_subset)
        qnet_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            qnet_raw, time_grid)
        vs.qnet = update(vs.qnet, at[2:-2, 2:-2, :], -qnet_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        qnec_raw = self._get_data(vs, 'dqdt', idx=data_subset)
        qnec_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            qnec_raw, time_grid)
        vs.qnec = update(vs.qnec, at[2:-2, 2:-2, :], qnec_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        qsol_raw = self._get_data(vs, 'swf', idx=data_subset)
        qsol_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                            qsol_raw, time_grid)
        vs.qsol = update(vs.qsol, at[2:-2, 2:-2, :], -qsol_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        # SST and SSS
        sst_raw = self._get_data(vs, 'sst', idx=data_subset)
        sst_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                           sst_raw, time_grid)
        vs.t_star = update(vs.t_star, at[2:-2, 2:-2, :], sst_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        sss_raw = self._get_data(vs, 'sss', idx=data_subset)
        sss_data = veros.tools.interpolate((xt_forc, yt_forc, np.arange(12)),
                                           sss_raw, time_grid)
        vs.s_star = update(vs.s_star, at[2:-2, 2:-2, :], sss_data * vs.maskT[2:-2, 2:-2, -1, np.newaxis])

        if settings.enable_idemix:
            tidal_energy_raw = self._get_data(vs, 'tidal_energy', idx=data_subset)
            tidal_energy_data = veros.tools.interpolate(
                (xt_forc, yt_forc), tidal_energy_raw, t_grid[:-1]
            )
            mask_x, mask_y = (i + 2 for i in np.indices((vs.nx, vs.ny)))
            mask_z = np.maximum(0, vs.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= vs.maskW[mask_x, mask_y, mask_z] / vs.rho_0
            vs.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = vs.zw / efold1_shortwave
        swarg2 = vs.zw / efold2_shortwave
        pen = rpart_shortwave * np.exp(swarg1) + (1.0 - rpart_shortwave) * np.exp(swarg2)
        pen = update(pen, at[-1], 0.)
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[1:], (pen[1:] - pen[:-1]) / vs.dzt[1:])
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[0], pen[0] / vs.dzt[0])

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.update(set_forcing_kernel(state))

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        diagnostics = state.diagnostics

        diagnostics['cfl_monitor'].output_frequency = settings.dt_tracer * 100
        diagnostics['tracer_monitor'].output_frequency = 86400.
        diagnostics['snapshot'].output_frequency = 86400.
        diagnostics['overturning'].output_frequency = 360 * 86400
        diagnostics['overturning'].sampling_frequency = 86400.
        diagnostics['energy'].output_frequency = 360 * 86400
        diagnostics['energy'].sampling_frequency = 86400.
        diagnostics['averages'].output_frequency = 10 * 86400
        diagnostics['averages'].sampling_frequency = 3600.

        average_vars = ['surface_taux', 'surface_tauy', 'forc_temp_surface', 'forc_salt_surface',
                        'psi', 'temp', 'salt', 'u', 'v', 'w', 'Nsqr', 'Hd', 'rho', 'kappaH']
        if settings.enable_skew_diffusion:
            average_vars += ['B1_gm', 'B2_gm']
        if settings.enable_TEM_friction:
            average_vars += ['kappa_gm', 'K_diss_gm']
        if settings.enable_tke:
            average_vars += ['tke', 'Prandtlnumber', 'mxl', 'tke_diss',
                             'forc_tke_surface', 'tke_surf_corr']
        if settings.enable_idemix:
            average_vars += ['E_iw', 'forc_iw_surface', 'iw_diss',
                             'c0', 'v0']
        if settings.enable_eke:
            average_vars += ['eke', 'K_gm', 'L_rossby', 'L_rhines']
        diagnostics['averages'].output_variables = average_vars

    @veros_routine
    def after_timestep(self, state):
        pass
