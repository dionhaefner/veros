from veros import veros_kernel
from veros.core.density import (
    gsw, linear_eq as lq, nonlinear_eq1 as nq1,
    nonlinear_eq2 as nq2, nonlinear_eq3 as nq3
)


@veros_kernel
def get_rho(state, salt_loc, temp_loc, press):
    """
    calculate density as a function of temperature, salinity and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return lq.linear_eq_of_state_rho(salt_loc, temp_loc)
    elif settings.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_rho(salt_loc, temp_loc)
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_rho(salt_loc, temp_loc, press)
    elif settings.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_rho(salt_loc, temp_loc)
    elif settings.eq_of_state_type == 5:
        return gsw.gsw_rho(salt_loc, temp_loc, press)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_potential_rho(state, salt_loc, temp_loc, press_ref=0.):
    """
    calculate potential density as a function of temperature, salinity
    and reference pressure

    Note:

        This is identical to get_rho for eq_of_state_type {1, 2, 4}

    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return lq.linear_eq_of_state_rho(salt_loc, temp_loc)
    elif settings.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_rho(salt_loc, temp_loc)
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_rho(salt_loc, temp_loc, press_ref)
    elif settings.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_rho(salt_loc, temp_loc)
    elif settings.eq_of_state_type == 5:
        return gsw.gsw_rho(salt_loc, temp_loc, press_ref)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_dyn_enthalpy(state, salt_loc, temp_loc, press):
    """
    calculate dynamic enthalpy as a function of temperature, salinity and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return lq.linear_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif settings.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif settings.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif settings.eq_of_state_type == 5:
        return gsw.gsw_dyn_enthalpy(salt_loc, temp_loc, press)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_salt(state, rho_loc, temp_loc, press_loc):
    """
    calculate salinity as a function of density, temperature and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return lq.linear_eq_of_state_salt(rho_loc, temp_loc)
    elif settings.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_salt(rho_loc, temp_loc)
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_salt(rho_loc, temp_loc, press_loc)
    elif settings.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_salt(rho_loc, temp_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_drhodT(state, salt_loc, temp_loc, press_loc):
    """
    calculate drho/dT as a function of temperature, salinity and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodT()
    elif settings.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodT(temp_loc)
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodT(temp_loc, press_loc)
    elif settings.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodT(temp_loc)
    elif settings.eq_of_state_type == 5:
        return gsw.gsw_drhodT(salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_drhodS(state, salt_loc, temp_loc, press_loc):
    """
    calculate drho/dS as a function of temperature, salinity and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodS()
    elif settings.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodS()
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodS()
    elif settings.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodS()
    elif settings.eq_of_state_type == 5:
        return gsw.gsw_drhodS(salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_drhodp(state, salt_loc, temp_loc, press_loc):
    """
    calculate drho/dP as a function of temperature, salinity and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodp()
    elif settings.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodp()
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodp(temp_loc)
    elif settings.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodp()
    elif settings.eq_of_state_type == 5:
        return gsw.gsw_drhodp(salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_int_drhodT(state, salt_loc, temp_loc, press_loc):
    """
    calculate int_z^0 drho/dT dz' as a function of temperature, salinity and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return press_loc * lq.linear_eq_of_state_drhodT()  # int_z^0rho_T dz = - rho_T z
    elif settings.eq_of_state_type == 2:
        return press_loc * nq1.nonlin1_eq_of_state_drhodT(temp_loc)
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_int_drhodT(temp_loc, press_loc)
    elif settings.eq_of_state_type == 4:
        return press_loc * nq3.nonlin3_eq_of_state_drhodT(temp_loc)
    elif settings.eq_of_state_type == 5:
        return -(1024.0 / 9.81) * gsw.gsw_dHdT(salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_kernel
def get_int_drhodS(state, salt_loc, temp_loc, press_loc):
    """
    calculate int_z^0 drho/dS dz' as a function of temperature, salinity and pressure
    """
    settings = state.settings

    if settings.eq_of_state_type == 1:
        return press_loc * lq.linear_eq_of_state_drhodS()  # int_z^0rho_T dz = - rho_T z
    elif settings.eq_of_state_type == 2:
        return press_loc * nq1.nonlin1_eq_of_state_drhodS()
    elif settings.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_int_drhodS(press_loc)
    elif settings.eq_of_state_type == 4:
        return press_loc * nq3.nonlin3_eq_of_state_drhodS()
    elif settings.eq_of_state_type == 5:
        return -(1024.0 / 9.81) * gsw.gsw_dHdS(salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')
