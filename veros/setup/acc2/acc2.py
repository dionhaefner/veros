#!/usr/bin/env python

import veros
import veros.tools

#from veros import Veros, veros_method, settings

yt_start = -39.0
yt_end = 43
yu_start = -40.0
yu_end = 42


class ACC2(veros.Veros):
    """
    A simple global model with a Southern Ocean and Atlantic part
    """
    @veros_method
    def set_parameter(self):
        self.identifier = "acc2"

        self.nx, self.ny, self.nz = 30, 42, 15
        self.dt_mom = 4800
        self.dt_tracer = 86400 / 2.
        self.runlen = 86400 * 365

        self.coord_degree = True
        self.enable_cyclic_x = True

        self.congr_epsilon = 1e-12
        self.congr_max_iterations = 5000

        self.enable_neutral_diffusion = True
        self.K_iso_0 = 1000.0
        self.K_iso_steep = 500.0
        self.iso_dslope = 0.005
        self.iso_slopec = 0.01
        self.enable_skew_diffusion = True

        self.enable_hor_friction = True
        self.A_h = (2 * self.degtom) ** 3 * 2e-11
        self.enable_hor_friction_cos_scaling = 1
        self.hor_friction_cosPower = 1

        self.enable_bottom_friction = True
        self.r_bot = 1e-5

        self.enable_implicit_vert_friction = True
        self.enable_tke = True
        self.c_k = 0.1
        self.c_eps = 0.7
        self.alpha_tke = 30.0
        self.mxl_min = 1e-8
        self.tke_mxl_choice = 2

        self.K_gm_0 = 1000.0

        self.enable_eke = True
        self.eke_k_max = 1e4
        self.eke_c_k = 0.4
        self.eke_c_eps = 0.5
        self.eke_cross = 2.
        self.eke_crhin = 1.0
        self.eke_lmin = 100.0
        self.enable_eke_superbee_advection = True
        self.enable_eke_isopycnal_diffusion = True

        self.enable_idemix = True
        self.enable_idemix_hor_diffusion = True
        self.enable_eke_diss_surfbot = True
        self.eke_diss_surfbot_frac = 0.2
        self.enable_idemix_superbee_advection = True

        self.eq_of_state_type = 3

    @veros_method
    def set_grid(self):
        ddz = [50., 70., 100., 140., 190., 240., 290., 340.,
               390., 440., 490., 540., 590., 640., 690.]
        self.dxt[:] = 2.0
        self.dyt[:] = 2.0
        self.x_origin = 0.0
        self.y_origin = -40.0
        self.dzt[:] = ddz[::-1]
        self.dzt[:] *= 1 / 2.5

    @veros_method
    def set_coriolis(self):
        self.coriolis_t[:, :] = 2 * self.omega * np.sin(self.yt[None, :] / 180. * self.pi)

    @veros_method
    def set_topography(self):
        (X, Y) = np.meshgrid(self.xt, self.yt)
        X = X.transpose()
        Y = Y.transpose()
        self.kbot[...] = (X > 1.0) | (Y < -20)

    @veros_method
    def set_initial_conditions(self):
        # initial conditions
        self.temp[:, :, :, 0:2] = ((1 - self.zt[None, None, :] / self.zw[0]) * 15 * self.maskT)[..., None]
        self.salt[:, :, :, 0:2] = 35.0 * self.maskT[..., None]

        # wind stress forcing
        taux = np.zeros(self.ny + 1, dtype=self.default_float_type)
        yt = self.yt[2:self.ny + 3]
        taux = (.1e-3 * np.sin(np.pi * (self.yu[2:self.ny + 3] - yu_start) / (-20.0 - yt_start))) * (yt < -20) \
            + (.1e-3 * (1 - np.cos(2 * np.pi * (self.yu[2:self.ny + 3] - 10.0) / (yu_end - 10.0)))) * (yt > 10)
        self.surface_taux[:, 2:self.ny + 3] = taux * self.maskU[:, 2:self.ny + 3, -1]

        # surface heatflux forcing
        self.t_star = 15 * np.invert((self.yt < -20) | (self.yt > 20)) \
            + 15 * (self.yt - yt_start) / (-20 - yt_start) * (self.yt < -20) \
            + 15 * (1 - (self.yt - 20) / (yt_end - 20)) * (self.yt > 20.)
        self.t_rest = self.dzt[np.newaxis, -1] / (30. * 86400.) * self.maskT[:, :, -1]

        if self.enable_tke:
            self.forc_tke_surface[2:-2, 2:-2] = np.sqrt((0.5 * (self.surface_taux[2:-2, 2:-2] + self.surface_taux[1:-3, 2:-2]))**2
                                                        + (0.5 * (self.surface_tauy[2:-2, 2:-2] + self.surface_tauy[2:-2, 1:-3]))**2)**(1.5)

        if self.enable_idemix:
            self.forc_iw_bottom[:] = 1.0e-6 * self.maskW[:, :, -1]
            self.forc_iw_surface[:] = 0.1e-6 * self.maskW[:, :, -1]

    @veros_method
    def set_forcing(self):
        self.forc_temp_surface[:] = self.t_rest * (self.t_star - self.temp[:, :, -1, self.tau])

    @veros_method
    def set_diagnostics(self):
        self.diagnostics["snapshot"].output_frequency = 86400 * 10
        self.diagnostics["averages"].output_variables = (
            "salt", "temp", "u", "v", "w", "psi", "surface_taux", "surface_tauy"
        )
        self.diagnostics["averages"].output_frequency = 365 * 86400.
        self.diagnostics["averages"].sampling_frequency = self.dt_tracer * 10
        self.diagnostics["overturning"].output_frequency = 365 * 86400. / 48.
        self.diagnostics["overturning"].sampling_frequency = self.dt_tracer * 10
        self.diagnostics["tracer_monitor"].output_frequency = 365 * 86400. / 12.
        self.diagnostics["energy"].output_frequency = 365 * 86400. / 48
        self.diagnostics["energy"].sampling_frequency = self.dt_tracer * 10

    def after_timestep(self):
        pass

@veros.tools.cli
def run(*args, **kwargs):
    simulation = ACC2(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()