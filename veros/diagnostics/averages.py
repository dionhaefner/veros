import os
import copy

from veros import veros_routine
from veros.diagnostics.diagnostic import VerosDiagnostic
from veros.variables import TIMESTEPS, allocate


class Averages(VerosDiagnostic):
    """Time average output diagnostic.

    All registered variables are summed up when :meth:`diagnose` is called,
    and averaged and output upon calling :meth:`output`.
    """
    name = 'averages' #:
    output_path = '{identifier}.averages.nc'  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_variables = None #: Iterable containing all variables to be averaged. Changes have no effect after ``initialize`` has been called.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.

    def initialize(self, state):
        """Register all variables to be averaged
        """
        self.average_nitts = 0
        self.average_vars = {}

        if not self.output_variables:
            return

        for var in self.output_variables:
            var_meta = copy.copy(state.var_meta[var])
            var_meta.time_dependent = True
            if self._has_timestep_dim(state, var):
                var_meta.dims = var_meta.dims[:-1]
            var_sum = allocate(state.dimensions, var_meta.dims)
            self.average_vars[var] = (var_meta, var_sum)

        self.initialize_output(state, {k: v[0] for k, v in self.average_vars.items()})

    @staticmethod
    def _has_timestep_dim(state, var):
        return state.var_meta[var].dims[-1] == TIMESTEPS[0]

    @veros_routine
    def diagnose(self, state):
        vs = state.variables

        self.average_nitts += 1
        for key, var in self.average_vars.items():
            var_meta, var_data = var
            if self._has_timestep_dim(state, key):
                var_data = var_data + getattr(vs, key)[..., vs.tau]
            else:
                var_data = var_data + getattr(vs, key)

            self.average_vars[key] = (var_meta, var_data)

    def output(self, state):
        """Write averages to netcdf file and zero array"""
        variable_metadata = {key: var[0] for key, var in self.average_vars.items()}
        variable_mean = {key: var[1] / self.average_nitts for key, var in self.average_vars.items()}

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state, variable_metadata)

        self.write_output(state, variable_metadata, variable_mean)

        for key, var in self.average_vars.items():
            var_meta, var_data = var
            self.average_vars[key] = (var_meta, 0 * var_data)

        self.average_nitts = 0

    def read_restart(self, state, infile):
        attributes, variables = self.read_h5_restart(state, state.var_meta, infile)

        if attributes:
            self.average_nitts = attributes['average_nitts']

        if variables:
            self.average_vars = variables

        for key, var in self.average_vars.items():
            var.time_dependent = True
            if self._has_timestep_dim(state, key):
                var.dims = var.dims[:-1]

    def write_restart(self, state, outfile):
        attributes = {'average_nitts': self.average_nitts}
        variables = {key: var[1] for key, var in self.average_vars.items()}
        variable_metadata = {key: var[0] for key, var in self.average_vars.items()}
        self.write_h5_restart(state, attributes, variable_metadata, variables, outfile)
