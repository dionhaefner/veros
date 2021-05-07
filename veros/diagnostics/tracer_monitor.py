from veros import logger

from veros.core.operators import numpy as np
from veros.diagnostics.diagnostic import VerosDiagnostic
from veros.distributed import global_sum


class TracerMonitor(VerosDiagnostic):
    """Diagnostic monitoring global tracer contents / fluxes.

    Writes output to stdout (no binary output).
    """
    name = 'tracer_monitor' #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    #: internal attributes to write to restart file
    restart_attributes = ('tempm1', 'vtemp1', 'saltm1', 'vsalt1')

    def initialize(self, state):
        self.tempm1 = 0.
        self.vtemp1 = 0.
        self.saltm1 = 0.
        self.vsalt1 = 0.

    def diagnose(self, state):
        pass

    def output(self, state):
        """
        Diagnose tracer content
        """
        vs = state.variables

        cell_volume = vs.area_t[2:-2, 2:-2, np.newaxis] * vs.dzt[np.newaxis, np.newaxis, :] \
            * vs.maskT[2:-2, 2:-2, :]
        volm = global_sum(np.sum(cell_volume))
        tempm = global_sum(np.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau]))
        saltm = global_sum(np.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau]))
        vtemp = global_sum(np.sum(cell_volume * vs.temp[2:-2, 2:-2, :, vs.tau]**2))
        vsalt = global_sum(np.sum(cell_volume * vs.salt[2:-2, 2:-2, :, vs.tau]**2))

        logger.diagnostic(' Mean temperature {:.2e} change to last {:.2e}'
                          .format(float(tempm / volm), float((tempm - self.tempm1) / volm)))
        logger.diagnostic(' Mean salinity    {:.2e} change to last {:.2e}'
                          .format(float(saltm / volm), float((saltm - self.saltm1) / volm)))
        logger.diagnostic(' Temperature var. {:.2e} change to last {:.2e}'
                          .format(float(vtemp / volm), float((vtemp - self.vtemp1) / volm)))
        logger.diagnostic(' Salinity var.    {:.2e} change to last {:.2e}'
                          .format(float(vsalt / volm), float((vsalt - self.vsalt1) / volm)))

        self.tempm1 = tempm
        self.vtemp1 = vtemp
        self.saltm1 = saltm
        self.vsalt1 = vsalt

    def read_restart(self, state, infile):
        attributes, _ = self.read_h5_restart(state, {}, infile)
        for attr in self.restart_attributes:
            setattr(self, attr, attributes[attr])

    def write_restart(self, state, outfile):
        attributes = {key: getattr(self, key) for key in self.restart_attributes}
        self.write_h5_restart(state, attributes, {}, {}, outfile)
