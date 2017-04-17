from collections import OrderedDict
import logging
import os

from . import io_tools
from .. import veros_class_method
from .diagnostic import VerosDiagnostic
from ..core import density
from ..variables import Variable

OVERTURNING_VARIABLES = OrderedDict([
    ("sigma", Variable(
        "Sigma axis", ("sigma",), "kg/m^3", "Sigma axis", output=True,
        time_dependent=False
    )),
    ("trans", Variable(
        "Meridional transport", ("yu", "sigma"), "m^3/s",
        "Meridional transport", output=True
    )),
    ("vsf_iso", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", output=True
    )),
    ("vsf_depth", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", output=True
    )),
])
ISONEUTRAL_VARIABLES = OrderedDict([
    ("bolus_iso", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", output=True
    )),
    ("bolus_depth", Variable(
        "Meridional transport", ("yu", "zw"), "m^3/s",
        "Meridional transport", output=True
    )),
])

class Overturning(VerosDiagnostic):
    """isopycnal overturning diagnostic
    """
    output_path = "{identifier}_overturning.nc"
    p_ref = 2000.

    @veros_class_method
    def initialize(self, veros):
        self.variables = OVERTURNING_VARIABLES
        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            self.variables.update(ISONEUTRAL_VARIABLES)

        self.nitts = 0
        self.nlevel = veros.nz * 4
        self._allocate(veros)

        # sigma levels
        self.sige = density.get_rho(veros, 35., -2., self.p_ref)
        self.sigs = density.get_rho(veros, 35., 30., self.p_ref)
        self.dsig = (self.sige - self.sigs) / (self.nlevel - 1)

        logging.debug(" sigma ranges for overturning diagnostic:")
        logging.debug(" start sigma0 = {:.1f}".format(self.sigs))
        logging.debug(" end sigma0 = {:.1f}".format(self.sige))
        logging.debug(" Delta sigma0 = {:.1e}".format(self.dsig))
        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            logging.debug(" also calculating overturning by eddy-driven velocities")

        self.sigma[...] = self.sigs + self.dsig * np.arange(self.nlevel)

        # precalculate area below z levels
        self.zarea[2:-2, :] = np.cumsum(np.sum(
                                  veros.dxt[2:-2, np.newaxis, np.newaxis] \
                                * veros.cosu[np.newaxis, 2:-2, np.newaxis] \
                                * veros.maskV[2:-2, 2:-2, :]
                             , axis=0) * veros.dzt[np.newaxis, :], axis=1)

        self.initialize_output(veros, self.variables,
                               var_data={"sigma": self.sigma},
                               extra_dimensions={"sigma": self.nlevel})

    @veros_class_method
    def _allocate(self, veros):
        self.sigma = np.zeros(self.nlevel)
        self.trans = np.zeros((veros.ny+4, self.nlevel))
        self.z_sig = np.zeros((veros.ny+4, self.nlevel))
        self.zarea = np.zeros((veros.ny+4, veros.nz))
        self.bolus_trans = np.zeros((veros.ny+4, self.nlevel))
        self.vsf_iso = np.zeros((veros.ny+4, veros.nz))
        self.vsf_depth = np.zeros((veros.ny+4, veros.nz))
        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            self.bolus_iso = np.zeros((veros.ny+4, veros.nz))
            self.bolus_depth = np.zeros((veros.ny+4, veros.nz))
        self.sig_loc = np.zeros((veros.nx+4, veros.ny+4, veros.nz))

        self.mean_variables = {
            "trans": np.zeros((veros.ny+4, self.nlevel)),
            "vsf_iso": np.zeros((veros.ny+4, veros.nz)),
            "vsf_depth": np.zeros((veros.ny+4, veros.nz))
        }
        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            self.mean_variables.update({
                "bolus_iso": np.zeros((veros.ny+4, veros.nz)),
                "bolus_depth": np.zeros((veros.ny+4, veros.nz))
            })


    @veros_class_method
    def diagnose(self, veros):
        # sigma at p_ref
        self.sig_loc[2:-2, 2:-1, :] = density.get_rho(veros,
                                  veros.salt[2:-2, 2:-1, :, veros.tau],
                                  veros.temp[2:-2, 2:-1, :, veros.tau],
                                  self.p_ref)

        # transports below isopycnals and area below isopycnals
        sig_loc_face = 0.5 * (self.sig_loc[2:-2, 2:-2, :] + self.sig_loc[2:-2, 3:-1, :])
        for m in range(self.nlevel):
            # NOTE: vectorized version would be O(N^4) in memory
            # consider cythonizing if performance-critical
            mask = sig_loc_face > self.sigma[m]
            self.trans[2:-2, m] = np.sum(
                                   veros.v[2:-2, 2:-2, :, veros.tau] \
                                 * veros.dxt[2:-2, np.newaxis, np.newaxis] \
                                 * veros.cosu[np.newaxis, 2:-2, np.newaxis] \
                                 * veros.maskV[2:-2, 2:-2, :] * mask
                               , axis=(0,2))
            self.z_sig[2:-2, m] = np.sum(
                                    veros.dzt[np.newaxis, np.newaxis, :] \
                                  * veros.dxt[2:-2, np.newaxis, np.newaxis] \
                                  * veros.cosu[np.newaxis, 2:-2, np.newaxis] \
                                  * veros.maskV[2:-2, 2:-2, :] * mask
                                , axis=(0,2))

        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            # eddy-driven transports below isopycnals
            for m in range(self.nlevel):
                # NOTE: see above
                mask = sig_loc_face > self.sigma[m]
                self.bolus_trans[2:-2, m] = np.sum(
                                                (veros.B1_gm[2:-2, 2:-2, 1:] \
                                               - veros.B1_gm[2:-2, 2:-2, :-1]) \
                                              * veros.dxt[2:-2, np.newaxis, np.newaxis] \
                                              * veros.cosu[np.newaxis, 2:-2, np.newaxis] \
                                              * veros.maskV[2:-2, 2:-2, 1:] \
                                              * mask[:, :, 1:]
                                            , axis=(0,2))
                self.bolus_trans[2:-2, m] += np.sum(
                                                veros.B1_gm[2:-2, 2:-2, 0] \
                                              * veros.dxt[2:-2, np.newaxis] \
                                              * veros.cosu[np.newaxis, 2:-2] \
                                              * veros.maskV[2:-2, 2:-2, 0] \
                                              * mask[:, :, 0]
                                            , axis=0)

        # streamfunction on geopotentials
        self.vsf_depth[2:-2, :] = np.cumsum(np.sum(
                                     veros.dxt[2:-2, np.newaxis, np.newaxis] \
                                   * veros.cosu[np.newaxis, 2:-2, np.newaxis] \
                                   * veros.v[2:-2, 2:-2, :, veros.tau] \
                                   * veros.maskV[2:-2, 2:-2, :]
                                 , axis=0) * veros.dzt[np.newaxis, :], axis=1)

        if veros.enable_neutral_diffusion and veros.enable_skew_diffusion:
            # streamfunction for eddy driven velocity on geopotentials
            self.bolus_depth[2:-2, :] = np.sum(
                                           veros.dxt[2:-2, np.newaxis, np.newaxis] \
                                         * veros.cosu[np.newaxis, 2:-2, np.newaxis] \
                                         * veros.B1_gm[2:-2, 2:-2, :] \
                                        , axis=0)
        # interpolate from isopycnals to depth
        self.vsf_iso[2:-2, :] = self._interpolate_along_axis(veros,
                                    self.z_sig[2:-2, :], self.trans[2:-2, :],
                                    self.zarea[2:-2, :], 1)
        self.bolus_iso[2:-2, :] = self._interpolate_along_axis(veros,
                                    self.z_sig[2:-2, :], self.bolus_trans[2:-2, :],
                                    self.zarea[2:-2, :], 1)

        # average in time
        self.nitts += 1
        for var, arr in self.mean_variables.items():
            arr[...] += getattr(self, var)

    @veros_class_method
    def _interpolate_along_axis(self, veros, coords, arr, interp_coords, axis=0):
        if coords.ndim == 1:
            if len(coords) != arr.shape[axis]:
                raise ValueError("Coordinate shape must match array shape along axis")
        elif coords.ndim == arr.ndim:
            if coords.shape != arr.shape:
                raise ValueError("Coordinate shape must match array shape")
        else:
            raise ValueError("Coordinate shape must match array dimensions")

        if axis != 0:
            arr = np.moveaxis(arr, axis, 0)
            coords = np.moveaxis(coords, axis, 0)
            interp_coords = np.moveaxis(interp_coords, axis, 0)

        diff = coords[np.newaxis, :, ...] - interp_coords[:, np.newaxis, ...]
        diff_m = np.where(diff <= 0, diff, np.inf)
        i_m = np.argmin(np.abs(diff_m), axis=1)
        i_p = np.minimum(coords.shape[0] - 1, i_m + 1)
        if coords.ndim == 1:
            full_shape = (slice(None),) + (np.newaxis,) * (arr.ndim - 1)
            i_p_full = i_p[full_shape] * np.ones(arr.shape)
            i_m_full = i_m[full_shape] * np.ones(arr.shape)
        else:
            i_p_full = i_p
            i_m_full = i_m
        ii = np.indices(interp_coords.shape)
        i_p_slice = (i_p,) + tuple(ii[1:])
        i_m_slice = (i_m,) + tuple(ii[1:])
        pos = np.where(i_p_full == i_m_full, 1., (coords[i_p_slice] - interp_coords) \
                        / (coords[i_p_slice] - coords[i_m_slice] + 1e-20))
        return np.moveaxis(arr[i_p_slice] * (1-pos) + arr[i_m_slice] * pos, 0, axis)

    @veros_class_method
    def output(self, veros):
        if not os.path.isfile(self.get_output_file_name(veros)):
            self.initialize_output(veros, self.variables,
                                   var_data={"sigma": self.sigma},
                                   extra_dimensions={"sigma": self.nlevel})

        if self.nitts > 0:
            for var, arr in self.mean_variables.items():
                arr[...] *= 1. / self.nitts

        var_metadata = {key: var for key, var in self.variables.items() if var.output}
        var_data = {key: getattr(self, key) for key, var in self.variables.items() if var.output}
        self.write_output(veros, var_metadata, var_data)

        self.nitts = 0
        for var, arr in self.mean_variables.items():
            arr[...] = 0.

    @veros_class_method
    def read_restart(self, veros):
        attributes, variables = self.read_h5_restart(veros)
        if attributes:
            self.nitts = attributes["nitts"]
        if variables:
            for var, arr in self.mean_variables.items():
                arr[...] = variables[var]

    @veros_class_method
    def write_restart(self, veros):
        self.write_h5_restart(veros, {"nitts": self.nitts}, self.mean_variables)