import importlib.util

from veros import VerosSetup, logger, settings, runtime_settings, runtime_state


def _load_fortran_module(module, path):
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LowercaseAttributeWrapper:
    """
    A simple wrapper class that converts attributes to lower case (needed for Fortran interface)
    """

    def __init__(self, wrapped_object):
        super().__setattr__(self, '_w', wrapped_object)

    def __getattr__(self, key):
        if key == '_w':
            return super().__getattribute__(self, '_w')
        return getattr(super().__getattribute__(self, '_w'), key.lower())

    def __setattr__(self, key, value):
        setattr(self._w, key.lower(), value)


class VerosLegacy(VerosSetup):
    """
    An alternative Veros class that supports the pyOM Fortran interface as backend

    .. warning::

       Do not use this class for new setups!

    """
    def __init__(self, pyom_lib=None, *args, **kwargs):
        """
        To use the pyOM2 legacy interface point the pyom_lib argument to the pyOM2 Fortran library:

        Example:

            >>> sim = GlobalOneDegreeSetup(pyom_lib='pyOM_code.so')
            >>> sim.setup()
            >>> sim.run()

        """
        super().__init__(*args, **kwargs)

        if pyom_lib:
            self.legacy_mode = True

            try:
                self.fortran = LowercaseAttributeWrapper(_load_fortran_module('pyOM_code', pyom_lib))
                self.use_mpi = False
            except ImportError:
                self.fortran = LowercaseAttributeWrapper(_load_fortran_module('pyOM_code_MPI', pyom_lib))
                self.use_mpi = True

                from mpi4py import MPI
                self.mpi_comm = MPI.COMM_WORLD

            self.main_module = LowercaseAttributeWrapper(self.fortran.main_module)
            self.isoneutral_module = LowercaseAttributeWrapper(self.fortran.isoneutral_module)
            self.idemix_module = LowercaseAttributeWrapper(self.fortran.idemix_module)
            self.tke_module = LowercaseAttributeWrapper(self.fortran.tke_module)
            self.eke_module = LowercaseAttributeWrapper(self.fortran.eke_module)

        else:
            self.legacy_mode = False
            self.use_mpi = False
            self.fortran = self
            self.main_module = self.state
            self.isoneutral_module = self.state
            self.idemix_module = self.state
            self.tke_module = self.state
            self.eke_module = self.state

        self.modules = (self.main_module, self.isoneutral_module, self.idemix_module,
                        self.tke_module, self.eke_module)

    def set_legacy_parameter(self):
        m = self.fortran.main_module
        m.n_pes_i, m.n_pes_j = runtime_settings.num_proc

        # define processor boundary idx (1-based)
        ipx, ipy = runtime_state.proc_idx
        m.is_pe = (m.nx // m.n_pes_i) * ipx + 1
        m.ie_pe = (m.nx // m.n_pes_i) * (ipx + 1)
        m.js_pe = (m.ny // m.n_pes_j) * ipy + 1
        m.je_pe = (m.ny // m.n_pes_j) * (ipy + 1)

        self.get_tau = lambda: m.tau - 1 if self.legacy_mode else m.tau

        # force settings that are not supported by Veros
        idm = self.fortran.idemix_module
        m.enable_streamfunction = True
        m.enable_hydrostatic = True
        idm.enable_idemix_m2 = False
        idm.enable_idemix_niw = False

    def _set_commandline_settings(self):
        for key, val in self.override_settings.items():
            for m in self.modules:
                if hasattr(m, key):
                    setattr(m, key, settings.SETTINGS[key].type(val))

    def setup(self, *args, **kwargs):
        vs = self.state
        with vs.timers['setup']:
            if self.legacy_mode:
                if self.use_mpi:
                    self.fortran.my_mpi_init(self.mpi_comm.py2f())
                else:
                    self.fortran.my_mpi_init(0)
                self.set_parameter(vs)
                self._set_commandline_settings()
                self.set_legacy_parameter()
                self.fortran.pe_decomposition()
                self.fortran.allocate_main_module()
                self.fortran.allocate_isoneutral_module()
                self.fortran.allocate_tke_module()
                self.fortran.allocate_eke_module()
                self.fortran.allocate_idemix_module()
                self.set_grid(vs)
                self.fortran.calc_grid()
                self.set_coriolis(vs)
                self.fortran.calc_beta()
                self.set_topography(vs)
                self.fortran.calc_topo()
                self.fortran.calc_spectral_topo()
                self.set_initial_conditions(vs)
                self.fortran.calc_initial_conditions()
                self.fortran.streamfunction_init()
                self.set_diagnostics(vs)
                self.set_forcing(vs)
                self.fortran.check_isoneutral_slope_crit()

            else:
                # self.set_parameter() is called twice, but that shouldn't matter
                self.set_parameter(vs)
                self._set_commandline_settings()
                self.set_legacy_parameter()
                super().setup(*args, **kwargs)

                diag_legacy_settings = (
                    (vs.diagnostics['cfl_monitor'], 'output_frequency', 'ts_monint'),
                    (vs.diagnostics['tracer_monitor'], 'output_frequency', 'trac_cont_int'),
                    (vs.diagnostics['snapshot'], 'output_frequency', 'snapint'),
                    (vs.diagnostics['averages'], 'output_frequency', 'aveint'),
                    (vs.diagnostics['averages'], 'sampling_frequency', 'avefreq'),
                    (vs.diagnostics['overturning'], 'output_frequency', 'overint'),
                    (vs.diagnostics['overturning'], 'sampling_frequency', 'overfreq'),
                    (vs.diagnostics['energy'], 'output_frequency', 'energint'),
                    (vs.diagnostics['energy'], 'sampling_frequency', 'energfreq'),
                )

                for diag, param, attr in diag_legacy_settings:
                    if hasattr(vs, attr):
                        setattr(diag, param, getattr(vs, attr))

    def run(self, **kwargs):
        if not self.legacy_mode:
            return super().run(**kwargs)

        vs = self.state
        f = self.fortran
        m = self.main_module
        idm = self.idemix_module
        ekm = self.eke_module
        tkm = self.tke_module

        logger.info('Starting integration for {:.2e}s'.format(float(m.runlen)))

        while vs.time < m.runlen:
            logger.info('Current iteration: {}'.format(m.itt))

            with vs.timers['main']:
                self.set_forcing(vs)

                if idm.enable_idemix:
                    f.set_idemix_parameter()

                f.set_eke_diffusivities()
                f.set_tke_diffusivities()

                with vs.timers['momentum']:
                    f.momentum()

                with vs.timers['temperature']:
                    f.thermodynamics()

                if ekm.enable_eke or tkm.enable_tke or idm.enable_idemix:
                    f.calculate_velocity_on_wgrid()

                with vs.timers['eke']:
                    if ekm.enable_eke:
                        f.integrate_eke()

                with vs.timers['idemix']:
                    if idm.enable_idemix:
                        f.integrate_idemix()

                with vs.timers['tke']:
                    if tkm.enable_tke:
                        f.integrate_tke()

                """
                Main boundary exchange
                for density, temp and salt this is done in integrate_tempsalt.f90
                """
                f.border_exchg_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe -
                                   m.onx, m.je_pe + m.onx, m.u[:, :, :, m.taup1 - 1], m.nz)
                f.setcyclic_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx,
                                m.je_pe + m.onx, m.u[:, :, :, m.taup1 - 1], m.nz)
                f.border_exchg_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe -
                                   m.onx, m.je_pe + m.onx, m.v[:, :, :, m.taup1 - 1], m.nz)
                f.setcyclic_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx,
                                m.je_pe + m.onx, m.v[:, :, :, m.taup1 - 1], m.nz)

                if tkm.enable_tke:
                    f.border_exchg_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe -
                                       m.onx, m.je_pe + m.onx, tkm.tke[:, :, :, m.taup1 - 1], m.nz)
                    f.setcyclic_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx,
                                    m.je_pe + m.onx, tkm.tke[:, :, :, m.taup1 - 1], m.nz)
                if ekm.enable_eke:
                    f.border_exchg_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe -
                                       m.onx, m.je_pe + m.onx, ekm.eke[:, :, :, m.taup1 - 1], m.nz)
                    f.setcyclic_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx,
                                    m.je_pe + m.onx, ekm.eke[:, :, :, m.taup1 - 1], m.nz)
                if idm.enable_idemix:
                    f.border_exchg_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe -
                                       m.onx, m.je_pe + m.onx, idm.e_iw[:, :, :, m.taup1 - 1], m.nz)
                    f.setcyclic_xyz(m.is_pe - m.onx, m.ie_pe + m.onx, m.js_pe - m.onx,
                                    m.je_pe + m.onx, idm.e_iw[:, :, :, m.taup1 - 1], m.nz)

                # diagnose vertical velocity at taup1
                f.vertical_velocity()

                # diagnose isoneutral streamfunction regardless of output settings
                f.isoneutral_diag_streamfunction()

            # shift time
            m.itt += 1
            vs.time += m.dt_tracer

            self.after_timestep(vs)

            otaum1 = int(m.taum1)
            m.taum1 = m.tau
            m.tau = m.taup1
            m.taup1 = otaum1

            # NOTE: benchmarks parse this, do not change / remove
            logger.debug('Time step took {}s', vs.timers['main'].get_last_time())

        logger.debug('Timing summary:')
        logger.debug(' setup time summary       = {}s', vs.timers['setup'].get_time())
        logger.debug(' main loop time summary   = {}s', vs.timers['main'].get_time())
        logger.debug('     momentum             = {}s', vs.timers['momentum'].get_time())
        logger.debug('     thermodynamics       = {}s', vs.timers['temperature'].get_time())
        logger.debug('     EKE                  = {}s', vs.timers['eke'].get_time())
        logger.debug('     IDEMIX               = {}s', vs.timers['idemix'].get_time())
        logger.debug('     TKE                  = {}s', vs.timers['tke'].get_time())
