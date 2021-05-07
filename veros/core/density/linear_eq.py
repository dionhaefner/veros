"""
==========================================================================
  linear equation of state
  input is Salinity sa in g/kg,
  pot. temperature ct in deg C
==========================================================================
"""
from veros import veros_kernel

rho0 = 1024.0
theta0 = 283.0 - 273.15
S0 = 35.0
betaT = 1.67e-4
betaS = 0.78e-3
grav = 9.81
z0 = 0.0


@veros_kernel
def linear_eq_of_state_rho(sa, ct):
    return -(betaT * (ct - theta0) - betaS * (sa - S0)) * rho0


@veros_kernel
def linear_eq_of_state_dyn_enthalpy(sa, ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return grav * zz * (-betaT * thetas + betaS * (sa - S0))


@veros_kernel
def linear_eq_of_state_salt(rho, ct):
    return (rho + betaT * (ct - theta0) * rho0) / (betaS * rho0) + S0


@veros_kernel
def linear_eq_of_state_drhodT():
    return -betaT * rho0


@veros_kernel
def linear_eq_of_state_drhodS():
    return betaS * rho0


@veros_kernel
def linear_eq_of_state_drhodp():
    return 0.0
