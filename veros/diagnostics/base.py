import abc
import os

from veros.io_tools import netcdf as nctools
from veros.signals import do_not_disturb
from veros import time, distributed, runtime_settings


class VerosDiagnostic(metaclass=abc.ABCMeta):
    """Base class for diagnostics. Provides an interface and wrappers for common I/O.

    Any diagnostic needs to implement the five interface methods and set some attributes.
    """
    name = None #: Name that identifies the current diagnostic
    sampling_frequency = 0.
    output_frequency = 0.
    output_path = None

    @abc.abstractmethod
    def initialize(self, state):
        """Called at the end of setup. Use this to process user settings and handle setup."""
        pass

    @abc.abstractmethod
    def diagnose(self, state):
        """Called with frequency ``sampling_frequency``."""
        pass

    @abc.abstractmethod
    def output(self, state):
        """Called with frequency ``output_frequency``."""
        pass

    @abc.abstractmethod
    def write_restart(self, state):
        """Responsible for writing restart files."""
        pass

    @abc.abstractmethod
    def read_restart(self, state):
        """Responsible for reading restart files."""
        pass

    def get_output_file_name(self, state):
        statedict = dict(state.variables.items())
        statedict.update(state.settings.items())
        return self.output_path.format(**statedict)

    @do_not_disturb
    def initialize_output(self, state, variables, var_data=None, extra_dimensions=None):
        if runtime_settings.diskless_mode or (not self.output_frequency and not self.sampling_frequency):
            return

        output_path = self.get_output_file_name(state)
        if os.path.isfile(output_path) and not runtime_settings.force_overwrite:
            raise IOError('output file {} for diagnostic "{}" exists '
                          '(change output path or enable force_overwrite runtime setting)'
                          .format(output_path, self.name))

        # possible race condition ahead!
        distributed.barrier()

        with nctools.threaded_io(state, output_path, 'w') as outfile:
            nctools.initialize_file(state, outfile)
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    nctools.add_dimension(state, dim_id, size, outfile)

            for key, var in variables.items():
                if key not in outfile.variables:
                    nctools.initialize_variable(state, key, var, outfile)

                if not var.time_dependent:
                    if var_data is None or key not in var_data:
                        raise ValueError('var_data argument must be given for constant variables')

                    nctools.write_variable(state, key, var, var_data[key], outfile)

    @do_not_disturb
    def write_output(self, state, variables, variable_data):
        vs = state.variables

        if runtime_settings.diskless_mode:
            return

        with nctools.threaded_io(state, self.get_output_file_name(state), 'r+') as outfile:
            time_step = nctools.get_current_timestep(state, outfile)
            current_days = time.convert_time(vs.time, 'seconds', 'days')
            nctools.advance_time(state, time_step, current_days, outfile)
            for key, var in variables.items():
                nctools.write_variable(state, key, var, variable_data[key],
                                       outfile, time_step=time_step)
