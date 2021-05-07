from veros.core.operators import numpy as np

from veros import veros_routine, veros_kernel, KernelOutput, runtime_settings
from veros.variables import allocate
from veros.core import friction, streamfunction
from veros.core.operators import update, update_add, at


@veros_kernel
def tend_coriolisf(state):
    """
    time tendency due to Coriolis force
    """
    vs = state.variables
    settings = state.settings

    vs.du_cor = update(vs.du_cor, at[2:-2, 2:-2], vs.maskU[2:-2, 2:-2] \
        * (vs.coriolis_t[2:-2, 2:-2, np.newaxis] * (vs.v[2:-2, 2:-2, :, vs.tau] + vs.v[2:-2, 1:-3, :, vs.tau])
           * vs.dxt[2:-2, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis]
            + vs.coriolis_t[3:-1, 2:-2, np.newaxis] *
           (vs.v[3:-1, 2:-2, :, vs.tau] + vs.v[3:-1, 1:-3, :, vs.tau])
           * vs.dxt[3:-1, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis]) * 0.25)
    vs.dv_cor = update(vs.dv_cor, at[2:-2, 2:-2], -1 * vs.maskV[2:-2, 2:-2] \
        * (vs.coriolis_t[2:-2, 2:-2, np.newaxis] * (vs.u[1:-3, 2:-2, :, vs.tau] + vs.u[2:-2, 2:-2, :, vs.tau])
           * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis]
           / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])
           + vs.coriolis_t[2:-2, 3:-1, np.newaxis]
           * (vs.u[1:-3, 3:-1, :, vs.tau] + vs.u[2:-2, 3:-1, :, vs.tau])
           * vs.dyt[np.newaxis, 3:-1, np.newaxis] * vs.cost[np.newaxis, 3:-1, np.newaxis]
           / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])) * 0.25)

    """
    time tendency due to metric terms
    """
    if settings.coord_degree:
        vs.du_cor = update_add(vs.du_cor, at[2:-2, 2:-2], vs.maskU[2:-2, 2:-2] * 0.125 * vs.tantr[np.newaxis, 2:-2, np.newaxis] \
            * ((vs.u[2:-2, 2:-2, :, vs.tau] + vs.u[1:-3, 2:-2, :, vs.tau])
               * (vs.v[2:-2, 2:-2, :, vs.tau] + vs.v[2:-2, 1:-3, :, vs.tau])
               * vs.dxt[2:-2, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis]
               + (vs.u[3:-1, 2:-2, :, vs.tau] + vs.u[2:-2, 2:-2, :, vs.tau])
               * (vs.v[3:-1, 2:-2, :, vs.tau] + vs.v[3:-1, 1:-3, :, vs.tau])
               * vs.dxt[3:-1, np.newaxis, np.newaxis] / vs.dxu[2:-2, np.newaxis, np.newaxis]))
        vs.dv_cor = update_add(vs.dv_cor, at[2:-2, 2:-2], -1 * vs.maskV[2:-2, 2:-2] * 0.125 \
            * (vs.tantr[np.newaxis, 2:-2, np.newaxis] * (vs.u[2:-2, 2:-2, :, vs.tau] + vs.u[1:-3, 2:-2, :, vs.tau])**2
               * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis]
               / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])
               + vs.tantr[np.newaxis, 3:-1, np.newaxis]
               * (vs.u[2:-2, 3:-1, :, vs.tau] + vs.u[1:-3, 3:-1, :, vs.tau])**2
               * vs.dyt[np.newaxis, 3:-1, np.newaxis] * vs.cost[np.newaxis, 3:-1, np.newaxis]
               / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])))

    """
    transfer to time tendencies
    """
    vs.du = update(vs.du, at[2:-2, 2:-2, :, vs.tau], vs.du_cor[2:-2, 2:-2])
    vs.dv = update(vs.dv, at[2:-2, 2:-2, :, vs.tau], vs.dv_cor[2:-2, 2:-2])

    return KernelOutput(du=vs.du, dv=vs.dv, du_cor=vs.du_cor, dv_cor=vs.dv_cor)


@veros_kernel
def tend_tauxyf(state):
    """
    wind stress forcing
    """
    vs = state.variables
    settings = state.settings

    if runtime_settings.pyom_compatibility_mode:
        vs.du = update_add(vs.du, at[2:-2, 2:-2, -1, vs.tau], vs.maskU[2:-2, 2:-2, -1] * vs.surface_taux[2:-2, 2:-2] / vs.dzt[-1])
        vs.dv = update_add(vs.dv, at[2:-2, 2:-2, -1, vs.tau], vs.maskV[2:-2, 2:-2, -1] * vs.surface_tauy[2:-2, 2:-2] / vs.dzt[-1])
    else:
        vs.du = update_add(vs.du, at[2:-2, 2:-2, -1, vs.tau], vs.maskU[2:-2, 2:-2, -1] * vs.surface_taux[2:-2, 2:-2] / vs.dzt[-1] / settings.rho_0)
        vs.dv = update_add(vs.dv, at[2:-2, 2:-2, -1, vs.tau], vs.maskV[2:-2, 2:-2, -1] * vs.surface_tauy[2:-2, 2:-2] / vs.dzt[-1] / settings.rho_0)

    return KernelOutput(du=vs.du, dv=vs.dv)


@veros_kernel
def momentum_advection(state):
    """
    Advection of momentum with second order which is energy conserving
    """
    vs = state.variables

    """
    Code from MITgcm
    """

    utr = vs.u[..., vs.tau] * vs.maskU * vs.dyt[np.newaxis, :, np.newaxis] \
        * vs.dzt[np.newaxis, np.newaxis, :]
    vtr = vs.dzt[np.newaxis, np.newaxis, :] * vs.cosu[np.newaxis, :, np.newaxis] \
        * vs.dxt[:, np.newaxis, np.newaxis] * vs.v[..., vs.tau] * vs.maskV
    wtr = vs.w[..., vs.tau] * vs.maskW * vs.area_t[:, :, np.newaxis]

    """
    for zonal momentum
    """
    flux_east = allocate(state.dimensions, ("xu", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yu", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yt", "zw"))

    flux_east = update(flux_east, at[1:-2, 2:-2], 0.25 * (vs.u[1:-2, 2:-2, :, vs.tau]
                                    + vs.u[2:-1, 2:-2, :, vs.tau]) \
                                 * (utr[2:-1, 2:-2] + utr[1:-2, 2:-2]))
    flux_north = update(flux_north, at[2:-2, 1:-2], 0.25 * (vs.u[2:-2, 1:-2, :, vs.tau]
                                     + vs.u[2:-2, 2:-1, :, vs.tau]) \
                                  * (vtr[3:-1, 1:-2] + vtr[2:-2, 1:-2]))
    flux_top = update(flux_top, at[2:-2, 2:-2, :-1], 0.25 * (vs.u[2:-2, 2:-2, 1:, vs.tau]
                                        + vs.u[2:-2, 2:-2, :-1, vs.tau]) \
                                     * (wtr[2:-2, 2:-2, :-1] + wtr[3:-1, 2:-2, :-1]))
    vs.du_adv = update(vs.du_adv, at[2:-2, 2:-2], -1 * vs.maskU[2:-2, 2:-2] * (flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2]
                                               + flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3]) \
                         / (vs.dzt[np.newaxis, np.newaxis, :] * vs.area_u[2:-2, 2:-2, np.newaxis]))

    tmp = -1 * vs.maskU / (vs.dzt * vs.area_u[:, :, np.newaxis])
    vs.du_adv = vs.du_adv + tmp * flux_top
    vs.du_adv = update_add(vs.du_adv, at[:, :, 1:], tmp[:, :, 1:] * -flux_top[:, :, :-1])

    """
    for meridional momentum
    """
    flux_top = update(flux_top, at[...], 0.)
    flux_east = update(flux_east, at[1:-2, 2:-2], 0.25 * (vs.v[1:-2, 2:-2, :, vs.tau]
                                    + vs.v[2:-1, 2:-2, :, vs.tau]) * (utr[1:-2, 3:-1] + utr[1:-2, 2:-2]))
    flux_north = update(flux_north, at[2:-2, 1:-2], 0.25 * (vs.v[2:-2, 1:-2, :, vs.tau]
                                     + vs.v[2:-2, 2:-1, :, vs.tau]) * (vtr[2:-2, 2:-1] + vtr[2:-2, 1:-2]))
    flux_top = update(flux_top, at[2:-2, 2:-2, :-1], 0.25 * (vs.v[2:-2, 2:-2, 1:, vs.tau]
                                        + vs.v[2:-2, 2:-2, :-1, vs.tau]) * (wtr[2:-2, 2:-2, :-1] + wtr[2:-2, 3:-1, :-1]))
    vs.dv_adv = update(vs.dv_adv, at[2:-2, 2:-2], -1 * vs.maskV[2:-2, 2:-2] * (flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2]
                                               + flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3]) \
                         / (vs.dzt * vs.area_u[2:-2, 2:-2, np.newaxis]))

    tmp = vs.dzt * vs.area_u[:, :, np.newaxis]
    vs.dv_adv = update_add(vs.dv_adv, at[:, :, 0], -1 * vs.maskV[:, :, 0] * flux_top[:, :, 0] / tmp[:, :, 0])
    vs.dv_adv = update_add(vs.dv_adv, at[:, :, 1:], -1 * vs.maskV[:, :, 1:] \
        * (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / tmp[:, :, 1:])

    vs.du = update_add(vs.du, at[:, :, :, vs.tau], vs.du_adv)
    vs.dv = update_add(vs.dv, at[:, :, :, vs.tau], vs.dv_adv)

    return KernelOutput(du=vs.du, dv=vs.dv, du_adv=vs.du_adv, dv_adv=vs.dv_adv)


@veros_routine
def vertical_velocity(state):
    vs = state.variables
    vs.update(vertical_velocity_kernel(state))


@veros_kernel
def vertical_velocity_kernel(state):
    """
    vertical velocity from continuity :
    \\int_0^z w_z dz = w(z)-w(0) = - \\int dz (u_x + v_y)
    w(z) = -int dz u_x + v_y
    """
    vs = state.variables

    fxa = allocate(state.dimensions, ("xt", "yt", "zw"))

    # integrate from bottom to surface to see error in w
    fxa = update(fxa, at[1:, 1:, 0], -1 * vs.maskW[1:, 1:, 0] * vs.dzt[0] * \
        ((vs.u[1:, 1:, 0, vs.taup1] - vs.u[:-1, 1:, 0, vs.taup1])
         / (vs.cost[np.newaxis, 1:] * vs.dxt[1:, np.newaxis])
         + (vs.cosu[np.newaxis, 1:] * vs.v[1:, 1:, 0, vs.taup1]
            - vs.cosu[np.newaxis, :-1] * vs.v[1:, :-1, 0, vs.taup1])
         / (vs.cost[np.newaxis, 1:] * vs.dyt[np.newaxis, 1:])))

    fxa = update(fxa, at[1:, 1:, 1:], -1 * vs.maskW[1:, 1:, 1:] * vs.dzt[np.newaxis, np.newaxis, 1:] \
        * ((vs.u[1:, 1:, 1:, vs.taup1] - vs.u[:-1, 1:, 1:, vs.taup1])
           / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis])
           + (vs.cosu[np.newaxis, 1:, np.newaxis] * vs.v[1:, 1:, 1:, vs.taup1]
              - vs.cosu[np.newaxis, :-1, np.newaxis] * vs.v[1:, :-1, 1:, vs.taup1])
           / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dyt[np.newaxis, 1:, np.newaxis])))

    # TODO: replace cumsum
    vs.w = update(vs.w, at[1:, 1:, :, vs.taup1], np.cumsum(fxa[1:, 1:, :], axis=2))

    return KernelOutput(w=vs.w)


@veros_routine
def momentum(state):
    """
    solve for momentum for taup1
    """
    vs = state.variables

    """
    time tendency due to Coriolis force
    """
    vs.update(tend_coriolisf(state))

    """
    wind stress forcing
    """
    vs.update(tend_tauxyf(state))

    """
    advection
    """
    vs.update(momentum_advection(state))

    with state.timers['friction']:
        friction.friction(state)

    """
    external mode
    """
    with state.timers['pressure']:
        streamfunction.solve_streamfunction(state)
