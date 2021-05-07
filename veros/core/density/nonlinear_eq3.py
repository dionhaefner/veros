"""
==========================================================================
  non-linear equation of state, no salinity dependency
  input is Salinity sa in g/kg,
  pot. temperature ct in deg C ,  no pressure dependency
==========================================================================
"""
from veros import veros_kernel

rho0 = 1024.0
theta0 = 283.0 - 273.15
S0 = 35.0
betaT = 1.67e-4
betaTs = 1e-5 / 2.
betaS = 0
grav = 9.81
z0 = 0.0


@veros_kernel
def nonlin3_eq_of_state_rho(sa, ct):
    thetas = ct - theta0
    return -(betaT * thetas + betaTs * thetas**2 - betaS * (sa - S0)) * rho0


@veros_kernel
def nonlin3_eq_of_state_dyn_enthalpy(sa, ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return grav * zz * (-betaT * thetas - betaTs * thetas**2 + betaS * (sa - S0))


@veros_kernel
def nonlin3_eq_of_state_salt(rho, ct):
    thetas = ct - theta0
    return (rho + (betaT * thetas + betaTs * thetas**2) * rho0) / (betaS * rho0) + S0


@veros_kernel
def nonlin3_eq_of_state_drhodT(ct):
    thetas = ct - theta0
    return -(betaT + 2 * betaTs * thetas) * rho0


@veros_kernel
def nonlin3_eq_of_state_drhodS():
    return betaS * rho0


@veros_kernel
def nonlin3_eq_of_state_drhodp():
    return 0.0
