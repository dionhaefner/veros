from veros import logger, runtime_settings, veros_kernel

from veros import time
from veros.diagnostics import averages, cfl_monitor, energy, overturning, snapshot, tracer_monitor
import veros.io_tools.hdf5 as h5tools


def create_default_diagnostics():
    return {Diag.name: Diag() for Diag in (averages.Averages, cfl_monitor.CFLMonitor,
                                           energy.Energy, overturning.Overturning,
                                           snapshot.Snapshot, tracer_monitor.TracerMonitor)}


@veros_kernel
def sanity_check(state):
    from veros.core.operators import numpy as np
    from veros.distributed import global_and
    return global_and(np.all(np.isfinite(state.variables.u)))


def read_restart(state):
    settings = state.settings
    if not settings.restart_input_filename:
        return

    if runtime_settings.force_overwrite:
        raise RuntimeError('To prevent data loss, force_overwrite cannot be used in restart runs')

    logger.info('Reading restarts')
    for diagnostic in state.diagnostics.values():
        diagnostic.read_restart(state, settings.restart_input_filename.format(**vars(state)))


def write_restart(state, force=False):
    vs = state.variables
    settings = state.settings

    if runtime_settings.diskless_mode:
        return

    if not settings.restart_output_filename:
        return

    if force or settings.restart_frequency and vs.time % settings.restart_frequency < settings.dt_tracer:
        statedict = dict(state.variables.items())
        statedict.update(state.settings.items())
        output_filename = settings.restart_output_filename.format(**statedict)
        logger.info(f'Writing restart file {output_filename} ...')

        with h5tools.threaded_io(state, output_filename, 'w') as outfile:
            for diagnostic in state.diagnostics.values():
                diagnostic.write_restart(state, outfile)


def initialize(state):
    for name, diagnostic in state.diagnostics.items():
        diagnostic.initialize(state)

        if diagnostic.sampling_frequency:
            logger.info(' Running diagnostic "{0}" every {1[0]:.1f} {1[1]}'
                         .format(name, time.format_time(diagnostic.sampling_frequency)))

        if diagnostic.output_frequency:
            logger.info(' Writing output for diagnostic "{0}" every {1[0]:.1f} {1[1]}'
                         .format(name, time.format_time(diagnostic.output_frequency)))


def diagnose(state):
    vs = state.variables
    settings = state.settings

    for diagnostic in state.diagnostics.values():
        if diagnostic.sampling_frequency and vs.time % diagnostic.sampling_frequency < settings.dt_tracer:
            diagnostic.diagnose(state)


def output(state):
    vs = state.variables
    settings = state.settings

    for diagnostic in state.diagnostics.values():
        if diagnostic.output_frequency and vs.time % diagnostic.output_frequency < settings.dt_tracer:
            diagnostic.output(state)
