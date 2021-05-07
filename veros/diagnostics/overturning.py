import os

from veros import logger, veros_routine, veros_kernel, KernelOutput

from veros.diagnostics.diagnostic import VerosDiagnostic
from veros.core import density
from veros.variables import Variable, allocate
from veros.distributed import global_sum
from veros.core.operators import numpy as np, update, update_add, at, for_loop


SIGMA = Variable(
    'Sigma axis', ('sigma',), 'kg/m^3', 'Sigma axis', output=True,
    time_dependent=False, write_to_restart=True
)

VARIABLES = {
    'trans': Variable(
        'Meridional transport', ('yu', 'sigma'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'vsf_iso': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'vsf_depth': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'bolus_iso': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
    'bolus_depth': Variable(
        'Meridional transport', ('yu', 'zw'), 'm^3/s',
        'Meridional transport', output=True, write_to_restart=True
    ),
}


@veros_kernel
def _interpolate_depth_coords(coords, arr, interp_coords):
    # ensure depth coordinates are monotonically increasing
    coords = -coords
    interp_coords = -interp_coords

    interp_vectorized = np.vectorize(np.interp, signature="(n),(m),(m)->(n)")
    return interp_vectorized(interp_coords, coords, arr)


@veros_kernel(static_args=("nlevel"))
def diagnose_kernel(state, nlevel, p_ref, sigma, zarea, vsf_depth, bolus_depth, vsf_iso, bolus_iso):
    vs = state.variables
    settings = state.settings

    # sigma at p_ref
    sig_loc = allocate(state.dimensions, ('xt', 'yt', 'zt'))
    sig_loc = update(sig_loc, at[2:-2, 2:-1, :], density.get_rho(state,
                                                vs.salt[2:-2, 2:-1, :, vs.tau],
                                                vs.temp[2:-2, 2:-1, :, vs.tau],
                                                p_ref)
    )

    # transports below isopycnals and area below isopycnals
    sig_loc_face = 0.5 * (sig_loc[2:-2, 2:-2, :] + sig_loc[2:-2, 3:-1, :])
    trans = allocate(state.dimensions, ('yu', nlevel))
    z_sig = allocate(state.dimensions, ('yu', nlevel))

    fac = (vs.dxt[2:-2, np.newaxis, np.newaxis]
            * vs.cosu[np.newaxis, 2:-2, np.newaxis]
            * vs.dzt[np.newaxis, np.newaxis, :]
            * vs.maskV[2:-2, 2:-2, :])

    def loop_body(m, values):
        trans, z_sig = values
        mask = sig_loc_face > sigma[m]
        trans = update(trans, at[2:-2, m], zonal_sum(np.sum(vs.v[2:-2, 2:-2, :, vs.tau] * fac * mask, axis=2)))
        z_sig = update(z_sig, at[2:-2, m], zonal_sum(np.sum(fac * mask, axis=2)))
        return (trans, z_sig)

    trans, z_sig = for_loop(0, nlevel, loop_body, init_val=(trans, z_sig))

    if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
        # eddy-driven transports below isopycnals
        bolus_trans = allocate(state.dimensions, ('yu', nlevel))

        def loop_body(m, bolus_trans):
            mask = sig_loc_face > sigma[m]
            bolus_trans = update(bolus_trans, at[2:-2, m], zonal_sum(
                np.sum(
                    (vs.B1_gm[2:-2, 2:-2, 1:] - vs.B1_gm[2:-2, 2:-2, :-1])
                    * vs.dxt[2:-2, np.newaxis, np.newaxis]
                    * vs.cosu[np.newaxis, 2:-2, np.newaxis]
                    * vs.maskV[2:-2, 2:-2, 1:]
                    * mask[:, :, 1:],
                    axis=2
                )

                + vs.B1_gm[2:-2, 2:-2, 0]
                * vs.dxt[2:-2, np.newaxis]
                * vs.cosu[np.newaxis, 2:-2]
                * vs.maskV[2:-2, 2:-2, 0]
                * mask[:, :, 0]
            ))
            return bolus_trans

        bolus_trans = for_loop(0, nlevel, loop_body, init_val=bolus_trans)

    # streamfunction on geopotentials
    vsf_depth = update_add(vsf_depth, at[2:-2, :], np.cumsum(zonal_sum(
        vs.dxt[2:-2, np.newaxis, np.newaxis]
        * vs.cosu[np.newaxis, 2:-2, np.newaxis]
        * vs.v[2:-2, 2:-2, :, vs.tau]
        * vs.maskV[2:-2, 2:-2, :]) * vs.dzt[np.newaxis, :], axis=1))

    if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
        # streamfunction for eddy driven velocity on geopotentials
        bolus_depth = update_add(bolus_depth, at[2:-2, :], zonal_sum(
            vs.dxt[2:-2, np.newaxis, np.newaxis]
            * vs.cosu[np.newaxis, 2:-2, np.newaxis]
            * vs.B1_gm[2:-2, 2:-2, :]))

    # interpolate from isopycnals to depth
    vsf_iso = update_add(vsf_iso, at[2:-2, :], _interpolate_depth_coords(
                                                            z_sig[2:-2, :], trans[2:-2, :],
                                                            zarea[2:-2, :]))
    if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
        bolus_iso = update_add(bolus_iso, at[2:-2, :], _interpolate_depth_coords(
                                                                z_sig[2:-2, :], bolus_trans[2:-2, :],
                                                                zarea[2:-2, :]))

    return KernelOutput(trans=trans, vsf_depth=vsf_depth, vsf_iso=vsf_iso, bolus_iso=bolus_iso, bolus_depth=bolus_depth)


def zonal_sum(arr):
    return global_sum(np.sum(arr, axis=0), axis=0)


class Overturning(VerosDiagnostic):
    """Isopycnal overturning diagnostic. Computes and writes vertical streamfunctions
    (zonally averaged).
    """

    name = 'overturning'  #:
    output_path = '{identifier}.overturning.nc'  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.
    p_ref = 2000.  #: Reference pressure for isopycnals

    def __init__(self):
        self.sigma_var = SIGMA
        self.mean_variables = VARIABLES
        self.variables = self.mean_variables.copy()
        self.variables.update({'sigma': self.sigma_var})

    def initialize(self, state):
        vs = state.variables
        settings = state.settings

        self.nitts = 0
        self.nlevel = settings.nz * 4

        self.sigma = allocate(state.dimensions, (self.nlevel,))
        self.zarea = allocate(state.dimensions, ('yu', 'zt'))
        self.trans = allocate(state.dimensions, ('yu', self.nlevel))
        self.vsf_iso = allocate(state.dimensions, ('yu', 'zt'))
        self.vsf_depth = allocate(state.dimensions, ('yu', 'zt'))

        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            self.bolus_iso = allocate(state.dimensions, ('yu', 'zt'))
            self.bolus_depth = allocate(state.dimensions, ('yu', 'zt'))

        # sigma levels
        self.sige = density.get_potential_rho(state, 35., -2., press_ref=self.p_ref)
        self.sigs = density.get_potential_rho(state, 35., 30., press_ref=self.p_ref)
        self.dsig = float(self.sige - self.sigs) / (self.nlevel - 1)

        logger.debug(' Sigma ranges for overturning diagnostic:')
        logger.debug(' Start sigma0 = {:.1f}'.format(self.sigs))
        logger.debug(' End sigma0 = {:.1f}'.format(self.sige))
        logger.debug(' Delta sigma0 = {:.1e}'.format(self.dsig))
        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            logger.debug(' Also calculating overturning by eddy-driven velocities')

        self.sigma = self.sigs + self.dsig * np.arange(self.nlevel)

        # precalculate area below z levels
        self.zarea = update(self.zarea, at[2:-2, :], np.cumsum(zonal_sum(
            vs.dxt[2:-2, np.newaxis, np.newaxis]
            * vs.cosu[np.newaxis, 2:-2, np.newaxis]
            * vs.maskV[2:-2, 2:-2, :]) * vs.dzt[np.newaxis, :], axis=1))

        self.initialize_output(state, self.variables,
                               var_data={'sigma': self.sigma},
                               extra_dimensions={'sigma': self.nlevel})

    @veros_routine
    def diagnose(self, state):
        res = diagnose_kernel(state, self.nlevel, self.p_ref, self.sigma, self.zarea, self.vsf_depth, self.bolus_depth, self.vsf_iso, self.bolus_iso)

        self.trans = self.trans + res.trans
        self.bolus_depth = res.bolus_depth
        self.bolus_iso = res.bolus_iso
        self.vsf_depth = res.vsf_depth
        self.vsf_iso = res.vsf_iso
        self.nitts += 1

    def output(self, state):
        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state, self.variables,
                                   var_data={'sigma': self.sigma},
                                   extra_dimensions={'sigma': self.nlevel})

        if self.nitts > 0:
            for var in self.mean_variables.keys():
                setattr(self, var, getattr(self, var) / self.nitts)

        var_metadata = {key: var for key, var in self.mean_variables.items()
                        if var.output and var.time_dependent}
        var_data = {key: getattr(self, key) for key, var in self.mean_variables.items()
                    if var.output and var.time_dependent}
        self.write_output(state, var_metadata, var_data)

        self.nitts = 0
        for var in self.mean_variables.keys():
            setattr(self, var, np.zeros_like(getattr(self, var)))

    def read_restart(self, state, infile):
        attributes, variables = self.read_h5_restart(state, self.variables, infile)
        if attributes:
            self.nitts = attributes['nitts']
        if variables:
            for var, arr in variables.items():
                getattr(self, var)[...] = arr

    def write_restart(self, state, outfile):
        var_data = {key: getattr(self, key)
                    for key, var in self.variables.items() if var.write_to_restart}
        self.write_h5_restart(state, {'nitts': self.nitts}, self.variables, var_data, outfile)
