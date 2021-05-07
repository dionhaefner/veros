import os

from veros import time, logger, veros_routine
from veros.diagnostics.diagnostic import VerosDiagnostic


class Snapshot(VerosDiagnostic):
    """Writes snapshots of the current solution. Also reads and writes the main restart
    data required for restarting a Veros simulation.
    """
    output_path = '{identifier}.snapshot.nc'
    """File to write to. May contain format strings that are replaced with Veros attributes."""
    name = 'snapshot' #:
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    #: Attributes to be written to restart file.
    restart_attributes = ('itt', 'time', 'tau', 'taum1', 'taup1')

    def initialize(self, state):
        vs = state.variables
        # TODO: initialize this earlier
        self.output_variables = [key for key, val in state.var_meta.items() if val.output and val.active]
        """Variables to be written to output. Defaults to all Veros variables that
        have the attribute :attr:`output`."""
        self.restart_variables = [key for key,
                                  val in state.var_meta.items() if val.write_to_restart and val.active]
        """Variables to be written to restart. Defaults to all Veros variables that
        have the attribute :attr:`write_to_restart`."""
        var_meta = {var: state.var_meta[var] for var in self.output_variables}
        var_data = {var: getattr(vs, var) for var in self.output_variables}
        self.initialize_output(state, var_meta, var_data)

    def diagnose(self, state):
        pass

    @veros_routine
    def output(self, state):
        vs = state.variables

        logger.info(' Writing snapshot at {0[0]:.2f} {0[1]}', time.format_time(vs.time))

        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize(state)

        var_meta = {var: state.var_meta[var]
                    for var in self.output_variables if state.var_meta[var].time_dependent}
        var_data = {var: getattr(vs, var) for var in var_meta.keys()}
        self.write_output(state, var_meta, var_data)

    def read_restart(self, state, infile):
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

    def write_restart(self, state, outfile):
        vs = state.variables
        restart_attributes = {key: getattr(vs, key) for key in self.restart_attributes}
        restart_vars = {var: state.var_meta[var] for var in self.restart_variables}
        restart_data = {var: getattr(vs, var) for var in self.restart_variables}
        self.write_h5_restart(state, restart_attributes, restart_vars, restart_data, outfile)
