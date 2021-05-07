import os

from veros import logger, runtime_settings, runtime_state
from veros.io_tools import hdf5 as h5tools
from veros.signals import do_not_disturb
from veros.distributed import get_chunk_slices, exchange_overlap
from veros.variables import get_shape


def read_h5_restart(dimensions, var_meta, restart_filename, groupname):
    from veros.core.operators import numpy as np, update, at

    if not os.path.isfile(restart_filename):
        raise IOError('restart file {} not found'.format(restart_filename))

    logger.info(' Reading restart data from {}', restart_filename)

    with h5tools.threaded_io(restart_filename, 'r') as infile:
        variables = {}

        for key, var in infile[groupname].items():
            if np.isscalar(var):
                variables[key] = var
                continue

            local_shape = get_shape(dimensions, var_meta[key].dims, local=True, include_ghosts=True)
            gidx, lidx = get_chunk_slices(var_meta[key].dims, include_overlap=True)

            variables[key] = np.empty(local_shape, dtype=str(var.dtype))
            variables[key] = update(variables[key], at[lidx], var[gidx])
            variables[key] = exchange_overlap(variables[key], var_meta[key].dims)

        attributes = {key: var.item() for key, var in infile[groupname].attrs.items()}

    return attributes, variables


@do_not_disturb
def write_h5_restart(dimensions, var_meta, var_data, restart_filename, groupname, extra_var_meta=None, attributes=None):
    with h5tools.threaded_io(restart_filename, 'r') as outfile:
        group = outfile.require_group(groupname)

        for key, var in var_data.items():
            global_shape = get_shape(dimensions, var.shape, var_meta[key].dims, local=False)
            gidx, lidx = get_chunk_slices(
                dimensions["xt"], dimensions["yt"], var_meta[key].dims, include_overlap=True)

            chunksize = []
            for d in var_meta[key].dims:
                if d in dimensions:
                    chunksize.append(get_shape(dimensions, (d,), local=True, include_ghosts=False)[0])
                else:
                    chunksize.append(1)

            kwargs = dict(
                exact=True,
                chunks=chunksize,
            )
            if runtime_settings.hdf5_gzip_compression and runtime_state.proc_num == 1:
                kwargs.update(
                    compression='gzip',
                    compression_opts=1
                )
            group.require_dataset(key, global_shape, var.dtype, **kwargs)
            group[key][gidx] = var[lidx]

        for key, val in attributes.items():
            group.attrs[key] = val


def read_core_restart(state):
    restart_vars = {var: state.variables[var] for var in self.restart_variables}
    restart_data = {var: getattr(state, var) for var in self.restart_variables}
    attributes, variables = self.read_h5_restart(state, restart_vars, infile)

    for key, arr in restart_data.items():
        try:
            restart_var = variables[key]

        except KeyError:
            logger.warning('Not reading restart data for variable {}: '
                            'no matching data found in restart file'
                            .format(key))
            continue

        if not arr.shape == restart_var.shape:
            logger.warning('Not reading restart data for variable {}: '
                            'restart data dimensions do not match model '
                            'grid'.format(key))
            continue

        arr[...] = restart_var

    for attr in self.restart_attributes:
        try:
            setattr(state, attr, attributes[attr])
        except KeyError:
            logger.warning('Not reading restart data for attribute {}: '
                            'attribute not found in restart file'
                            .format(attr))


def write_core_restart(state):
    vs = state.variables
    restart_attributes = {key: getattr(vs, key) for key in self.restart_attributes}
    restart_vars = {var: state.var_meta[var] for var in self.restart_variables}
    restart_data = {var: getattr(vs, var) for var in self.restart_variables}
    write_h5_restart(state, restart_attributes, restart_vars, restart_data, outfile)


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
