
import abc

# do not import veros.core here!
from veros import (
    settings, time, signals, distributed, progress,
    runtime_settings as rs, logger
)
from veros.state import get_default_state
from veros.plugins import load_plugin
from veros.routines import veros_routine, record_routine_stack, is_veros_routine


class VerosSetup(metaclass=abc.ABCMeta):
    """Main class for Veros, used for building a model and running it.

    Note:
        This class is meant to be subclassed. Subclasses need to implement the
        methods :meth:`set_parameter`, :meth:`set_topography`, :meth:`set_grid`,
        :meth:`set_coriolis`, :meth:`set_initial_conditions`, :meth:`set_forcing`,
        and :meth:`set_diagnostics`.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from veros import VerosSetup
        >>>
        >>> class MyModel(VerosSetup):
        >>>     ...
        >>>
        >>> simulation = MyModel()
        >>> simulation.run()
        >>> plt.imshow(simulation.state.variables.psi[..., 0])
        >>> plt.show()

    """
    __veros_plugins__ = tuple()

    def __init__(self, override=None):
        self.override_settings = override or {}

        # this should be the first time the core routines are imported
        import veros.core  # noqa: F401

        self._plugin_interfaces = tuple(load_plugin(p) for p in self.__veros_plugins__)
        self._setup_done = False
        self._routine_stack = None
        self._setup_routine_stack = None

        self.state = get_default_state(use_plugins=self.__veros_plugins__)

    @abc.abstractmethod
    def set_parameter(self, state):
        """To be implemented by subclass.

        First function to be called during setup.
        Use this to modify the model settings.

        Example:
          >>> def set_parameter(self, state):
          >>>     settings = state.settings
          >>>     settings.nx, settings.ny, settings.nz = (360, 120, 50)
          >>>     settings.coord_degree = True
          >>>     settings.enable_cyclic = True
        """
        pass

    @abc.abstractmethod
    def set_initial_conditions(self, state):
        """To be implemented by subclass.

        May be used to set initial conditions.

        Example:
          >>> @veros_method
          >>> def set_initial_conditions(self, state):
          >>>     vs = state.variables
          >>>     vs.u = update(vs.u, at[:, :, :, vs.tau], np.random.rand(vs.u.shape[:-1]))
        """
        pass

    @abc.abstractmethod
    def set_grid(self, state):
        """To be implemented by subclass.

        Has to set the grid spacings :attr:`dxt`, :attr:`dyt`, and :attr:`dzt`,
        along with the coordinates of the grid origin, :attr:`x_origin` and
        :attr:`y_origin`.

        Example:
          >>> @veros_method
          >>> def set_grid(self, state):
          >>>     vs = state.variables
          >>>     vs.x_origin, vs.y_origin = 0, 0
          >>>     vs.dxt = [0.1, 0.05, 0.025, 0.025, 0.05, 0.1]
          >>>     vs.dyt = 1.
          >>>     vs.dzt = [10, 10, 20, 50, 100, 200]
        """
        pass

    @abc.abstractmethod
    def set_coriolis(self, state):
        """To be implemented by subclass.

        Has to set the Coriolis parameter :attr:`coriolis_t` at T grid cells.

        Example:
          >>> @veros_method
          >>> def set_coriolis(self, state):
          >>>     vs = state.variables
          >>>     vs.coriolis_t = 2 * vs.omega * np.sin(vs.yt[np.newaxis, :] / 180. * vs.pi)
        """
        pass

    @abc.abstractmethod
    def set_topography(self, state):
        """To be implemented by subclass.

        Must specify the model topography by setting :attr:`kbot`.

        Example:
          >>> @veros_method
          >>> def set_topography(self, state):
          >>>     vs = state.variables
          >>>     vs.kbot = update(vs.kbot, at[...], 10)
          >>>     # add a rectangular island somewhere inside the domain
          >>>     vs.kbot = update(vs.kbot, at[10:20, 10:20], 0)
        """
        pass

    @abc.abstractmethod
    def set_forcing(self, state):
        """To be implemented by subclass.

        Called before every time step to update the external forcing, e.g. through
        :attr:`forc_temp_surface`, :attr:`forc_salt_surface`, :attr:`surface_taux`,
        :attr:`surface_tauy`, :attr:`forc_tke_surface`, :attr:`temp_source`, or
        :attr:`salt_source`. Use this method to implement time-dependent forcing.

        Example:
          >>> @veros_method
          >>> def set_forcing(self, state):
          >>>     vs = state.variables
          >>>     current_month = (vs.time / (31 * 24 * 60 * 60)) % 12
          >>>     vs.surface_taux = vs._windstress_data[:, :, current_month]
        """
        pass

    @abc.abstractmethod
    def set_diagnostics(self, vs):
        """To be implemented by subclass.

        Called before setting up the :ref:`diagnostics <diagnostics>`. Use this method e.g. to
        mark additional :ref:`variables <variables>` for output.

        Example:
          >>> @veros_method
          >>> def set_diagnostics(self, state):
          >>>     state.diagnostics['snapshot'].output_vars += ['drho', 'dsalt', 'dtemp']
        """
        pass

    @abc.abstractmethod
    def after_timestep(self, state):
        """Called at the end of each time step. Can be used to define custom, setup-specific
        events.
        """
        pass

    def _ensure_setup_done(self):
        if not self._setup_done:
            raise RuntimeError('setup() method has to be called before running the model')

    def setup(self):
        from veros import diagnostics
        from veros.core import numerics, streamfunction, eke, isoneutral

        setup_funcs = (
            self.set_parameter, self.set_grid, self.set_coriolis,
            self.set_topography, self.set_initial_conditions, self.set_diagnostics,
            self.set_forcing
        )

        for f in setup_funcs:
            if not is_veros_routine(f):
                raise RuntimeError(
                    f"{f.__name__} method is not a Veros routine. Please make sure to decorate it "
                    "with @veros_routine and try again."
                )

        logger.info('Running model setup')

        with self.state.settings.unlock():
            self.set_parameter(self.state)

            for setting, value in self.override_settings.items():
                setattr(self.state.settings, setting, value)

        settings.check_setting_conflicts(self.state.settings)
        distributed.validate_decomposition(self.state.dimensions)

        self.state.initialize_variables()

        with self.state.timers['setup'], record_routine_stack() as recorded_stack:
            self.set_grid(self.state)
            numerics.calc_grid(self.state)

            self.set_coriolis(self.state)
            numerics.calc_beta(self.state)

            self.set_topography(self.state)
            numerics.calc_topo(self.state)

            self.set_initial_conditions(self.state)
            numerics.calc_initial_conditions(self.state)
            streamfunction.streamfunction_init(self.state)
            eke.init_eke(self.state)

            for plugin in self._plugin_interfaces:
                plugin.setup_entrypoint(self.state)

            self.set_diagnostics(self.state)
            diagnostics.initialize(self.state)
            diagnostics.read_restart(self.state)

            self.set_forcing(self.state)
            isoneutral.check_isoneutral_slope_crit(self.state)

        self._setup_done = True
        self._setup_routine_stack = recorded_stack

    @veros_routine
    def step(self, state):
        from veros import diagnostics
        from veros.core import (
            idemix, eke, tke, momentum, thermodynamics, advection, utilities, isoneutral
        )

        self._ensure_setup_done()

        vs = state.variables
        settings = state.settings

        with state.profile_timers['all']:
            with state.timers['diagnostics']:
                diagnostics.write_restart(state)

            with state.timers['main']:
                with state.timers['forcing']:
                    self.set_forcing(state)

                if state.settings.enable_idemix:
                    with state.timers['idemix']:
                        idemix.set_idemix_parameter(state)

                with state.timers['eke']:
                    eke.set_eke_diffusivities(state)

                with state.timers['tke']:
                    tke.set_tke_diffusivities(state)

                with state.timers['momentum']:
                    momentum.momentum(state)

                with state.timers['thermodynamics']:
                    thermodynamics.thermodynamics(state)

                if settings.enable_eke or settings.enable_tke or settings.enable_idemix:
                    with state.timers['advection']:
                        advection.calculate_velocity_on_wgrid(state)

                with state.timers['eke']:
                    if state.settings.enable_eke:
                        eke.integrate_eke(state)

                with state.timers['idemix']:
                    if state.settings.enable_idemix:
                        idemix.integrate_idemix(state)

                with state.timers['tke']:
                    if state.settings.enable_tke:
                        tke.integrate_tke(state)

                with state.timers['boundary_exchange']:
                    vs.u = utilities.enforce_boundaries(vs.u, settings.enable_cyclic_x)
                    vs.v = utilities.enforce_boundaries(vs.v, settings.enable_cyclic_x)
                    if settings.enable_tke:
                        vs.tke = utilities.enforce_boundaries(vs.tke, settings.enable_cyclic_x)
                    if settings.enable_eke:
                        vs.eke = utilities.enforce_boundaries(vs.eke, settings.enable_cyclic_x)
                    if settings.enable_idemix:
                        vs.E_iw = utilities.enforce_boundaries(vs.E_iw, settings.enable_cyclic_x)

                with state.timers['momentum']:
                    momentum.vertical_velocity(state)

            with state.timers['plugins']:
                for plugin in self._plugin_interfaces:
                    with state.timers[plugin.name]:
                        plugin.run_entrypoint(state)

            vs.itt = vs.itt + 1
            vs.time = vs.time + settings.dt_tracer

            self.after_timestep(state)

            with state.timers['diagnostics']:
                if not diagnostics.sanity_check(state):
                    raise RuntimeError('solution diverged at iteration {}'.format(vs.itt))

                isoneutral.isoneutral_diag_streamfunction(state)
                diagnostics.diagnose(state)
                diagnostics.output(state)

            # NOTE: benchmarks parse this, do not change / remove
            logger.debug(' Time step took {:.2f}s', state.timers['main'].get_last_time())

            # permutate time indices
            vs.taum1, vs.tau, vs.taup1 = vs.tau, vs.taup1, vs.taum1

    def run(self, show_progress_bar=None):
        """Main routine of the simulation.

        Note:
            Make sure to call :meth:`setup` prior to this function.

        Arguments:
            show_progress_bar (:obj:`bool`, optional): Whether to show fancy progress bar via tqdm.
                By default, only show if stdout is a terminal and Veros is running on a single process.

        """
        from veros import diagnostics
        self._ensure_setup_done()

        vs = self.state.variables
        settings = self.state.settings

        logger.info('\nStarting integration for {0[0]:.1f} {0[1]}'.format(time.format_time(settings.runlen)))

        start_time = vs.time

        pbar = progress.get_progress_bar(self.state, use_tqdm=show_progress_bar)
        first_iteration = True

        try:
            with signals.signals_to_exception(), pbar:
                while vs.time - start_time < settings.runlen:
                    if first_iteration:
                        with record_routine_stack() as recorded_stack:
                            self.step(self.state)
                            self._routine_stack = recorded_stack

                        first_iteration = False
                    else:
                        self.step(self.state)

                    pbar.advance_time(settings.dt_tracer)

        except:  # noqa: E722
            logger.critical('Stopping integration at iteration {}', vs.itt)
            raise

        else:
            logger.success('Integration done\n')

        finally:
            diagnostics.write_restart(self.state, force=True)
            self._timing_summary()

    def _timing_summary(self):
        timing_summary = []

        timing_summary.extend([
            '',
            'Timing summary:',
            ' setup time               = {:.2f}s'.format(self.state.timers['setup'].get_time()),
            ' main loop time           = {:.2f}s'.format(self.state.timers['main'].get_time()),
            '   forcing                = {:.2f}s'.format(self.state.timers['forcing'].get_time()),
            '   momentum               = {:.2f}s'.format(self.state.timers['momentum'].get_time()),
            '     pressure             = {:.2f}s'.format(self.state.timers['pressure'].get_time()),
            '     friction             = {:.2f}s'.format(self.state.timers['friction'].get_time()),
            '   thermodynamics         = {:.2f}s'.format(self.state.timers['thermodynamics'].get_time()),
        ])

        if rs.profile_mode:
            timing_summary.extend([
                '     lateral mixing       = {:.2f}s'.format(self.state.timers['isoneutral'].get_time()),
                '     vertical mixing      = {:.2f}s'.format(self.state.timers['vmix'].get_time()),
                '     equation of state    = {:.2f}s'.format(self.state.timers['eq_of_state'].get_time()),
            ])

        timing_summary.extend([
            '   advection              = {:.2f}s'.format(self.state.timers['advection'].get_time()),
            '   EKE                    = {:.2f}s'.format(self.state.timers['eke'].get_time()),
            '   IDEMIX                 = {:.2f}s'.format(self.state.timers['idemix'].get_time()),
            '   TKE                    = {:.2f}s'.format(self.state.timers['tke'].get_time()),
            '   boundary exchange      = {:.2f}s'.format(self.state.timers['boundary_exchange'].get_time()),
            ' diagnostics and I/O      = {:.2f}s'.format(self.state.timers['diagnostics'].get_time()),
            ' plugins                  = {:.2f}s'.format(self.state.timers['plugins'].get_time()),
        ])

        timing_summary.extend([
            '   {:<22} = {:.2f}s'.format(plugin.name, self.state.timers[plugin.name].get_time())
            for plugin in self.state._plugin_interfaces
        ])

        logger.debug('\n'.join(timing_summary))

        if rs.profile_mode:
            print_profile_summary(self.state.profile_timers)


def print_profile_summary(profile_timers):
    profile_timings = ['', 'Profile timings:', '(total time spent)', '---']
    maxwidth = max(len(k) for k in profile_timers.keys())
    profile_format_string = '{{:<{}}} = {{:.2f}}s ({{:.2f}}%)'.format(maxwidth)
    total_time = max(profile_timers['all'].get_time(), 1e-8)  # prevent division by 0

    for name, timer in profile_timers.items():
        this_time = timer.get_time()
        profile_timings.append(
            profile_format_string.format(name, this_time, 100 * this_time / total_time)
        )

    logger.diagnostic('\n'.join(profile_timings))
