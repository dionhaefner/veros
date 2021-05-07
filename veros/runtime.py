import os
from collections import namedtuple

from veros.backend import BACKENDS
from veros.logs import LOGLEVELS


# MPI helpers

def _default_mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    else:
        return MPI.COMM_WORLD


# validators

def parse_two_ints(v):
    return (int(v[0]), int(v[1]))


def parse_choice(choices, preserve_case=False):
    def validate(choice):
        if isinstance(choice, str) and not preserve_case:
            choice = choice.lower()

        if choice not in choices:
            raise ValueError('must be one of {}'.format(choices))

        return choice

    return validate


def parse_bool(obj):
    if not isinstance(obj, str):
        return bool(obj)

    return obj.lower() in {'1', 'true', 'on'}


def check_mpi_comm(comm):
    if comm is not None:
        from mpi4py import MPI

        if not isinstance(comm, MPI.Comm):
            raise TypeError('mpi_comm must be Comm instance or None')

    return comm


def set_loglevel(loglevel):
    from veros import logs, runtime_settings as rs
    loglevel = parse_choice(LOGLEVELS)(loglevel)
    logs.setup_logging(loglevel=loglevel, log_all_processes=rs.log_all_processes)
    return loglevel


def set_log_all_processes(log_all_processes):
    from veros import logs, runtime_settings as rs
    log_all_processes = parse_bool(log_all_processes)
    logs.setup_logging(loglevel=rs.loglevel, log_all_processes=log_all_processes)
    return log_all_processes


DEVICES = ('cpu', 'gpu', 'tpu')
FLOAT_TYPES = ('float64', 'float32')
LINEAR_SOLVERS = ('scipy', 'petsc', 'best')


# settings

RuntimeSetting = namedtuple('RuntimeSetting', ('type', 'default', 'read_from_env'))
RuntimeSetting.__new__.__defaults__ = (None, None, True)

AVAILABLE_SETTINGS = {
    'backend': RuntimeSetting(parse_choice(BACKENDS), 'numpy'),
    'device': RuntimeSetting(parse_choice(DEVICES), 'cpu'),
    'float_type': RuntimeSetting(parse_choice(FLOAT_TYPES), 'float64'),
    'linear_solver': RuntimeSetting(parse_choice(LINEAR_SOLVERS), 'best'),
    'num_proc': RuntimeSetting(parse_two_ints, (1, 1), read_from_env=False),
    'profile_mode': RuntimeSetting(parse_bool, False),
    'loglevel': RuntimeSetting(set_loglevel, 'info'),
    'mpi_comm': RuntimeSetting(check_mpi_comm, _default_mpi_comm(), read_from_env=False),
    'log_all_processes': RuntimeSetting(set_log_all_processes, False),
    'use_io_threads': RuntimeSetting(parse_bool, False),
    'io_timeout': RuntimeSetting(float, 20),
    'hdf5_gzip_compression': RuntimeSetting(bool, True),
    'force_overwrite': RuntimeSetting(bool, False),
    'diskless_mode': RuntimeSetting(bool, False),
    'pyom_compatibility_mode': RuntimeSetting(bool, False),

}


class RuntimeSettings:
    __slots__ = [
        '__locked__',
        '__setting_types__',
        '__settings__',
        *AVAILABLE_SETTINGS.keys()
    ]

    def __init__(self, **kwargs):
        self.__locked__ = False
        self.__setting_types__ = {}

        for name, setting in AVAILABLE_SETTINGS.items():
            setting_envvar = 'VEROS_{}'.format(name.upper())

            if name in kwargs:
                val = kwargs[name]
            elif setting.read_from_env:
                val = os.environ.get(setting_envvar, setting.default)
            else:
                val = setting.default

            self.__setting_types__[name] = setting.type
            super().__setattr__(name, val)

        self.__settings__ = set(self.__setting_types__.keys())

    def update(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

        return self

    def __setattr__(self, attr, val):
        if getattr(self, '__locked__', False):
            raise RuntimeError('Runtime settings cannot be modified after import of core modules')

        if attr.startswith('_'):
            return super(RuntimeSettings, self).__setattr__(attr, val)

        # coerce type
        stype = self.__setting_types__.get(attr)
        if stype is not None:
            try:
                val = stype(val)
            except (TypeError, ValueError) as e:
                raise ValueError('Got invalid value for runtime setting "{}": {}'.format(attr, str(e))) from None

        return super(RuntimeSettings, self).__setattr__(attr, val)

    def __repr__(self):
        setval = ', '.join(
            '%s=%s' % (key, repr(getattr(self, key))) for key in self.__settings__
        )
        return '{clsname}({setval})'.format(
            clsname=self.__class__.__name__,
            setval=setval
        )


# state

class RuntimeState:
    """Unifies attributes from various modules in a simple read-only object"""
    __slots__ = ()

    @property
    def proc_rank(self):
        from veros import runtime_settings
        comm = runtime_settings.mpi_comm

        if comm is None:
            return 0

        return comm.Get_rank()

    @property
    def proc_num(self):
        from veros import runtime_settings
        comm = runtime_settings.mpi_comm

        if comm is None:
            return 1

        return comm.Get_size()

    @property
    def proc_idx(self):
        from veros import distributed
        return distributed.proc_rank_to_index(self.proc_rank)

    @property
    def backend_module(self):
        from veros import backend, runtime_settings
        return backend.get_backend_module(runtime_settings.backend)

    def __setattr__(self, attr, val):
        raise TypeError('Cannot modify {} objects'.format(self.__class__.__name__))
