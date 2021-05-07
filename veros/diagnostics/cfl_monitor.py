from veros import logger, veros_routine
from veros.core.operators import numpy as np
from veros.diagnostics.diagnostic import VerosDiagnostic
from veros.distributed import global_max


class CFLMonitor(VerosDiagnostic):
    """Diagnostic monitoring the maximum CFL number of the solution to detect
    instabilities.

    Writes output to stdout (no binary output).
    """
    name = 'cfl_monitor' #:
    output_frequency = None  # :Frequency (in seconds) in which output is written.

    def initialize(self, state):
        pass

    @veros_routine
    def diagnose(self, state):
        pass

    def output(self, state):
        """
        check for CFL violation
        """
        vs = state.variables
        settings = state.settings

        cfl = global_max(max(
            np.max(np.abs(vs.u[2:-2, 2:-2, :, vs.tau]) * vs.maskU[2:-2, 2:-2, :]
                   / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                   * vs.dt_tracer),
            np.max(np.abs(vs.v[2:-2, 2:-2, :, vs.tau]) * vs.maskV[2:-2, 2:-2, :]
                   / vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.dt_tracer)
        ))
        wcfl = global_max(np.max(
            np.abs(vs.w[2:-2, 2:-2, :, vs.tau]) * vs.maskW[2:-2, 2:-2, :]
                      / vs.dzt[np.newaxis, np.newaxis, :] * vs.dt_tracer
        ))

        if np.isnan(cfl) or np.isnan(wcfl):
            raise RuntimeError('CFL number is NaN at iteration {}'.format(vs.itt))

        logger.diagnostic(' Maximal hor. CFL number = {}'.format(float(cfl)))
        logger.diagnostic(' Maximal ver. CFL number = {}'.format(float(wcfl)))

        if settings.enable_eke or settings.enable_tke or settings.enable_idemix:
            cfl = global_max(max(
                np.max(np.abs(vs.u_wgrid[2:-2, 2:-2, :]) * vs.maskU[2:-2, 2:-2, :]
                       / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                       * vs.dt_tracer),
                np.max(np.abs(vs.v_wgrid[2:-2, 2:-2, :]) * vs.maskV[2:-2, 2:-2, :]
                       / vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.dt_tracer)
            ))
            wcfl = global_max(np.max(
                np.abs(vs.w_wgrid[2:-2, 2:-2, :]) * vs.maskW[2:-2, 2:-2, :]
                    / vs.dzt[np.newaxis, np.newaxis, :] * vs.dt_tracer
            ))
            logger.diagnostic(' Maximal hor. CFL number on w grid = {}'.format(float(cfl)))
            logger.diagnostic(' Maximal ver. CFL number on w grid = {}'.format(float(wcfl)))

    def read_restart(self, state, infile):
        pass

    def write_restart(self, state, outfile):
        pass
