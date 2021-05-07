from math import pi
from collections import namedtuple

Setting = namedtuple('setting', ('default', 'type', 'description'))

SETTINGS = {
    'identifier': Setting('UNNAMED', str, 'Identifier of the current simulation'),

    # Model parameters
    'nx': Setting(0, int, 'Grid points in zonal (x) direction'),
    'ny': Setting(0, int, 'Grid points in meridional (y,j) direction'),
    'nz': Setting(0, int, 'Grid points in vertical (z,k) direction'),
    'dt_mom': Setting(0., float, 'Time step in seconds for momentum'),
    'dt_tracer': Setting(0., float, 'Time step for tracers, can be larger than dt_mom'),
    'runlen': Setting(0., float, 'Length of simulation in seconds'),
    'AB_eps': Setting(0.1, float, 'Deviation from Adam-Bashforth weighting'),
    'x_origin': Setting(0, float, 'Grid origin in x-direction'),
    'y_origin': Setting(0, float, 'Grid origin in y-direction'),

    # Physical constants
    'pi': Setting(pi, float, 'Pi'),
    'radius': Setting(6370e3, float, 'Earth radius in m'),
    'degtom': Setting(6370e3 / 180. * pi, float, 'Conversion factor from degrees latitude to meters'),
    'omega': Setting(pi / 43082., float, 'Earth rotation frequency in 1/s'),
    'rho_0': Setting(1024., float, 'Boussinesq reference density in :math:`kg/m^3`'),
    'grav': Setting(9.81, float, 'Gravitational constant in :math:`m/s^2`'),

    # Logical switches for general model setup
    'coord_degree': Setting(False, bool, 'either spherical (True) or cartesian (False) coordinates'),
    'enable_cyclic_x': Setting(False, bool, 'enable cyclic boundary conditions'),
    'eq_of_state_type': Setting(1, int, 'equation of state: 1: linear, 3: nonlinear with comp., 5: TEOS'),
    'enable_implicit_vert_friction': Setting(False, bool, 'enable implicit vertical friction'),
    'enable_explicit_vert_friction': Setting(False, bool, 'enable explicit vertical friction'),
    'enable_hor_friction': Setting(False, bool, 'enable horizontal friction'),
    'enable_hor_diffusion': Setting(False, bool, 'enable horizontal diffusion'),
    'enable_biharmonic_friction': Setting(False, bool, 'enable biharmonic horizontal friction'),
    'enable_biharmonic_mixing': Setting(False, bool, 'enable biharmonic horizontal mixing'),
    'enable_hor_friction_cos_scaling': Setting(False, bool, 'scaling of hor. viscosity with cos(latitude)**cosPower'),
    'enable_ray_friction': Setting(False, bool, 'enable Rayleigh damping'),
    'enable_bottom_friction': Setting(False, bool, 'enable bottom friction'),
    'enable_bottom_friction_var': Setting(False, bool, 'enable bottom friction with lateral variations'),
    'enable_quadratic_bottom_friction': Setting(False, bool, 'enable quadratic bottom friction'),
    'enable_tempsalt_sources': Setting(False, bool, 'enable restoring zones, etc'),
    'enable_momentum_sources': Setting(False, bool, 'enable restoring zones, etc'),
    'enable_superbee_advection': Setting(False, bool, 'enable advection scheme with implicit mixing'),
    'enable_conserve_energy': Setting(True, bool, 'exchange energy consistently'),
    'enable_store_bottom_friction_tke': Setting(False, bool, 'transfer dissipated energy by bottom/rayleig fric. to TKE, else transfer to internal waves'),
    'enable_store_cabbeling_heat': Setting(False, bool, 'transfer non-linear mixing terms to potential enthalpy, else transfer to TKE and EKE'),
    'enable_noslip_lateral': Setting(False, bool, 'enable lateral no-slip boundary conditions in harmonic- and biharmonic friction.'),

    # External mode
    'congr_epsilon': Setting(1e-12, float, 'convergence criteria for Poisson solver'),
    'congr_max_iterations': Setting(1000, int, 'maximum number of Poisson solver iterations'),

    # Mixing parameters
    'A_h': Setting(0.0, float, 'lateral viscosity in m^2/s'),
    'K_h': Setting(0.0, float, 'lateral diffusivity in m^2/s'),
    'r_ray': Setting(0.0, float, 'Rayleigh damping coefficient in 1/s'),
    'r_bot': Setting(0.0, float, 'bottom friction coefficient in 1/s'),
    'r_quad_bot': Setting(0.0, float, 'qudratic bottom friction coefficient'),
    'hor_friction_cosPower': Setting(3, float, ''),
    'A_hbi': Setting(0.0, float, 'lateral biharmonic viscosity in m^4/s'),
    'K_hbi': Setting(0.0, float, 'lateral biharmonic diffusivity in m^4/s'),
    'kappaH_0': Setting(0.0, float, ''),
    'kappaM_0': Setting(0.0, float, 'fixed values for vertical viscosity/diffusivity which are set for no TKE model'),

    # Options for isopycnal mixing
    'enable_neutral_diffusion': Setting(False, bool, 'enable isopycnal mixing'),
    'enable_skew_diffusion': Setting(False, bool, 'enable skew diffusion approach for eddy-driven velocities'),
    'enable_TEM_friction': Setting(False, bool, 'TEM approach for eddy-driven velocities'),
    'K_iso_0': Setting(0.0, float, 'constant for isopycnal diffusivity in m^2/s'),
    'K_iso_steep': Setting(0.0, float, 'lateral diffusivity for steep slopes in m^2/s'),
    'K_gm_0': Setting(0.0, float, 'fixed value for K_gm which is set for no EKE model'),
    'iso_dslope': Setting(0.0008, float, 'parameters controlling max allowed isopycnal slopes'),
    'iso_slopec': Setting(0.001, float, 'parameters controlling max allowed isopycnal slopes'),

    # Idemix 1.0
    'enable_idemix': Setting(False, bool, ''),
    'tau_v': Setting(2.0 * 86400.0, float, 'time scale for vertical symmetrisation'),
    'tau_h': Setting(15.0 * 86400.0, float, 'time scale for horizontal symmetrisation'),
    'gamma': Setting(1.57, float, ''),
    'jstar': Setting(5.0, float, 'spectral bandwidth in modes'),
    'mu0': Setting(1. / 3., float, 'dissipation parameter'),
    'enable_idemix_hor_diffusion': Setting(False, bool, ''),
    'enable_eke_diss_bottom': Setting(False, bool, ''),
    'enable_eke_diss_surfbot': Setting(False, bool, ''),
    'eke_diss_surfbot_frac': Setting(1.0, float, 'fraction which goes into bottom'),
    'enable_idemix_superbee_advection': Setting(False, bool, ''),
    'enable_idemix_upwind_advection': Setting(False, bool, ''),

    # TKE
    'enable_tke': Setting(False, bool, ''),
    'c_k': Setting(0.1, float, ''),
    'c_eps': Setting(0.7, float, ''),
    'alpha_tke': Setting(1.0, float, ''),
    'mxl_min': Setting(1e-12, float, ''),
    'kappaM_min': Setting(0., float, ''),
    'kappaM_max': Setting(100., float, ''),
    'tke_mxl_choice': Setting(1, int, ''),
    'enable_tke_superbee_advection': Setting(False, bool, ''),
    'enable_tke_upwind_advection': Setting(False, bool, ''),
    'enable_tke_hor_diffusion': Setting(False, bool, ''),
    'K_h_tke': Setting(2000., float, 'lateral diffusivity for tke'),

    # EKE
    'enable_eke': Setting(False, bool, ''),
    'eke_lmin': Setting(100.0, float, 'minimal length scale in m'),
    'eke_c_k': Setting(1.0, float, ''),
    'eke_cross': Setting(1.0, float, 'Parameter for EKE model'),
    'eke_crhin': Setting(1.0, float, 'Parameter for EKE model'),
    'eke_c_eps': Setting(1.0, float, 'Parameter for EKE model'),
    'eke_k_max': Setting(1e4, float, 'maximum of K_gm'),
    'alpha_eke': Setting(1.0, float, 'factor vertical friction'),
    'enable_eke_superbee_advection': Setting(False, bool, ''),
    'enable_eke_upwind_advection': Setting(False, bool, ''),
    'enable_eke_isopycnal_diffusion': Setting(False, bool, 'use K_gm also for isopycnal diffusivity'),

    'enable_eke_leewave_dissipation': Setting(False, bool, ''),
    'c_lee0': Setting(1., float, ''),
    'eke_Ri0': Setting(200., float, ''),
    'eke_Ri1': Setting(50., float, ''),
    'eke_int_diss0': Setting(1. / (20 * 86400.), float, ''),
    'kappa_EKE0': Setting(0.1, float, ''),
    'eke_r_bot': Setting(0.0, float, 'bottom friction coefficient'),
    'eke_hrms_k0_min': Setting(0.0, float, 'min value for bottom roughness parameter'),

    # Restarts
    'restart_input_filename': Setting('', str, 'File name of restart input. If not given, no restart data will be read.'),
    'restart_output_filename': Setting('{identifier}_{itt:0>4d}.restart.h5', str, 'File name of restart output. May contain Python format syntax that is substituted with Veros attributes.'),
    'restart_frequency': Setting(0, float, 'Frequency (in seconds) to write restart data'),

    # New
    'kappaH_min': Setting(0., float, 'minimum value for vertical diffusivity'),
    'enable_kappaH_profile': Setting(False, bool, 'Compute vertical profile of diffusivity after Bryan and Lewis (1979) in TKE routine'),
    'enable_Prandtl_tke': Setting(True, bool, 'Compute Prandtl number from stratification levels in TKE routine'),
    'Prandtl_tke0': Setting(10., float, 'Constant Prandtl number when stratification is neglected for kappaH computation in TKE routine'),
}


def check_setting_conflicts(settings):
    if settings.enable_tke and not settings.enable_implicit_vert_friction:
        raise RuntimeError('use TKE model only with implicit vertical friction'
                           '(set enable_implicit_vert_fricton to True)')
