"""
==========================================================================
  non-linear equation of state from Vallis 2008
  input is Salinity sa in g/kg,
  pot. temperature ct in deg C and
  pressure p in dbar
==========================================================================
"""
from veros import veros_kernel

rho0 = 1024.0
z0 = 0.0
theta0 = 283.0 - 273.15
S0 = 35.0
grav = 9.81
cs0 = 1490.0
betaT = 1.67e-4
betaTs = 1e-5
betaS = 0.78e-3
gammas = 1.1e-8


@veros_kernel
def nonlin2_eq_of_state_rho(sa, ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return -(grav * zz / cs0**2 + betaT * (1 - gammas * grav * zz * rho0) * thetas + betaTs / 2 * thetas**2 - betaS * (sa - S0)) * rho0


@veros_kernel
def nonlin2_eq_of_state_dyn_enthalpy(sa, ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return grav * 0.5 * zz**2 * (-grav / cs0**2 + betaT * grav * rho0 * gammas * thetas) + grav * zz * (-betaT * thetas - betaTs * thetas**2 + betaS * (sa - S0))


@veros_kernel
def nonlin2_eq_of_state_salt(rho, ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return (rho / rho0 + (grav * zz / cs0**2 + betaT * (1 - gammas * grav * zz * rho0) * thetas + betaTs / 2 * thetas**2)) / betaS + S0


@veros_kernel
def nonlin2_eq_of_state_drhodT(ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return -(betaTs * thetas + betaT * (1 - gammas * grav * zz * rho0)) * rho0


@veros_kernel
def nonlin2_eq_of_state_drhodS():
    return betaS * rho0


@veros_kernel
def nonlin2_eq_of_state_drhodP(ct):
    thetas = ct - theta0
    return 1 / cs0**2 - betaT * gammas * rho0 * thetas


@veros_kernel
def nonlin2_eq_of_state_int_drhodT(ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return rho0 * zz * (betaT + betaTs * thetas) - rho0 * betaT * gammas * grav * rho0 * zz**2 / 2


@veros_kernel
def nonlin2_eq_of_state_int_drhodS(p):
    zz = -p - z0
    return -betaS * rho0 * zz
