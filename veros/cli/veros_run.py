import functools
import inspect
import os
import sys
import time
import importlib

import click

from veros import (
    runtime_settings, runtime_state, VerosSetup,
    __version__ as veros_version
)
from veros.settings import SETTINGS
from veros.backend import BACKENDS
from veros.runtime import LOGLEVELS, DEVICES, FLOAT_TYPES


class VerosSetting(click.ParamType):
    name = 'setting'
    current_key = None

    def convert(self, value, param, ctx):
        assert param.nargs == 2

        if self.current_key is None:
            if value not in SETTINGS:
                self.fail('Unknown setting %s' % value)
            self.current_key = value
            return value

        assert self.current_key in SETTINGS
        setting = SETTINGS[self.current_key]
        self.current_key = None

        if setting.type is bool:
            return click.BOOL(value)

        return setting.type(value)


def _import_from_file(path):
    module = os.path.basename(path).split(".py")[0]
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(setup_file, *args, slave, **kwargs):
    """Runs a Veros setup from given file"""
    total_proc = kwargs['num_proc'][0] * kwargs['num_proc'][1]

    if total_proc > 1 and runtime_state.proc_num == 1 and not slave:
        from mpi4py import MPI

        comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=['-m', 'mpi4py'] + list(sys.argv) + ['--slave'],
            maxprocs=total_proc
        )

        futures = [comm.irecv(source=p) for p in range(total_proc)]
        while True:
            done, success = zip(*(f.test() for f in futures))

            if not all(success):
                raise RuntimeError('An MPI worker encountered an error')

            if all(done):
                break

            time.sleep(0.1)

        return

    kwargs['override'] = dict(kwargs['override'])

    runtime_setting_kwargs = (
        'backend', 'profile_mode', 'num_proc', 'loglevel', 'device', 'float_type',
        'diskless_mode', 'force_overwrite',
    )
    for setting in runtime_setting_kwargs:
        setattr(runtime_settings, setting, kwargs.pop(setting))

    # determine setup class from given Python file
    setup_module = _import_from_file(setup_file)

    SetupClass = None
    for obj in vars(setup_module).values():
        if inspect.isclass(obj) and issubclass(obj, VerosSetup) and obj is not VerosSetup:
            if SetupClass is not None:
                raise RuntimeError("Veros setups can only contain one VerosSetup class")

            SetupClass = obj

    from veros import logger
    target_version = getattr(setup_module, '__VEROS_VERSION__', None)
    if target_version and target_version != veros_version:
        logger.warning(
            f"This is Veros v{veros_version}, but the given setup was generated with v{target_version}. "
            "Consider switching to this version of Veros or updating your setup file.\n"
        )

    try:
        sim = SetupClass(*args, **kwargs)
        sim.setup()
        sim.run()
    except:  # noqa: E722
        status = False
        raise
    else:
        status = True
    finally:
        if slave:
            runtime_settings.mpi_comm.Get_parent().send(status, dest=0)


@click.command('veros-run')
@click.argument("SETUP_FILE", type=click.Path(readable=True, dir_okay=False, resolve_path=True))
@click.option('-b', '--backend', default='numpy', type=click.Choice(BACKENDS),
                help='Backend to use for computations', show_default=True)
@click.option('--device', default='cpu', type=click.Choice(DEVICES),
                help='Hardware device to use (JAX backend only)', show_default=True)
@click.option('-v', '--loglevel', default='info', type=click.Choice(LOGLEVELS),
                help='Log level used for output', show_default=True)
@click.option('-s', '--override', nargs=2, multiple=True, metavar='SETTING VALUE',
                type=VerosSetting(), default=tuple(),
                help='Override model setting, may be specified multiple times')
@click.option('-p', '--profile-mode', is_flag=True, default=False, type=click.BOOL, envvar='VEROS_PROFILE',
                help='Write a performance profile for debugging', show_default=True)
@click.option('--force-overwrite', is_flag=True, help='Silently overwrite existing outputs')
@click.option('--diskless-mode', is_flag=True, help='Supress all output to disk')
@click.option('--float-type', default='float64', type=click.Choice(FLOAT_TYPES),
                help='Floating point precision to use', show_default=True)
@click.option('-n', '--num-proc', nargs=2, default=[1, 1], type=click.INT,
                help='Number of processes in x and y dimension')
@click.option('--slave', default=False, is_flag=True, hidden=True,
                help='Indicates that this process is an MPI worker (for internal use)')
@functools.wraps(run)
def cli(setup_file, *args, **kwargs):
    if not setup_file.endswith(".py"):
        raise click.Abort(f"The given setup file {setup_file} does not appear to be a Python file.")

    return run(setup_file, *args, **kwargs)


if __name__ == "__main__":
    cli()
