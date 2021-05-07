from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, KernelOutput, runtime_settings
from veros.variables import allocate
from veros.core import advection, utilities
from veros.core.operators import update, update_add, at

"""
IDEMIX as in Olbers and Eden, 2013
"""


@veros_kernel
def set_idemix_parameter(state):
    """
    set main IDEMIX parameter
    """
    vs = state.variables
    settings = state.settings

    bN0 = np.sum(np.sqrt(np.maximum(0., vs.Nsqr[:, :, :-1, vs.tau]))
                 * vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[:, :, :-1], axis=2) \
        + np.sqrt(np.maximum(0., vs.Nsqr[:, :, -1, vs.tau])) \
        * 0.5 * vs.dzw[-1:] * vs.maskW[:, :, -1]
    fxa = np.sqrt(np.maximum(0., vs.Nsqr[..., vs.tau])) \
        / (1e-22 + np.abs(vs.coriolis_t[..., np.newaxis]))

    cstar = np.maximum(1e-2, bN0[:, :, np.newaxis] / (settings.pi * settings.jstar))

    vs.c0 = np.maximum(0., settings.gamma * cstar * gofx2(fxa, settings.pi) * vs.maskW)
    vs.v0 = np.maximum(0., settings.gamma * cstar * hofx1(fxa, settings.pi) * vs.maskW)
    vs.alpha_c = np.maximum(1e-4, settings.mu0 * np.arccosh(np.maximum(1., fxa)) * np.abs(vs.coriolis_t[..., np.newaxis]) / cstar**2) * vs.maskW

    return KernelOutput(c0=vs.c0, v0=vs.v0, alpha_c=vs.alpha_c)


@veros_kernel
def integrate_idemix_kernel(state):
    """
    integrate idemix on W grid
    """
    vs = state.variables
    settings = state.settings

    a_tri, b_tri, c_tri, d_tri, delta = (allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2] for _ in range(5))
    forc = allocate(state.dimensions, ("xt", "yt", "zt"))
    maxE_iw = allocate(state.dimensions, ("xt", "yt", "zt"))

    """
    forcing by EKE dissipation
    """
    if settings.enable_eke:
        forc = vs.eke_diss_iw

    else:  # shortcut without EKE model
        if settings.enable_store_cabbeling_heat:
            forc = vs.K_diss_h - vs.P_diss_skew - vs.P_diss_hmix - vs.P_diss_iso

        else:
            forc = vs.K_diss_h - vs.P_diss_skew

        if settings.enable_TEM_friction:
            forc = forc + vs.K_diss_gm

    if settings.enable_eke and (settings.enable_eke_diss_bottom or settings.enable_eke_diss_surfbot):
        """
        vertically integrate EKE dissipation and inject at bottom and/or surface
        """
        a_loc = np.sum(vs.dzw[np.newaxis, np.newaxis, :-1] *
                       forc[:, :, :-1] * vs.maskW[:, :, :-1], axis=2)
        a_loc += 0.5 * forc[:, :, -1] * vs.maskW[:, :, -1] * vs.dzw[-1]

        forc = np.zeros_like(forc)

        ks = np.maximum(0, vs.kbot[2:-2, 2:-2] - 1)
        mask = ks[:, :, np.newaxis] == np.arange(settings.nz)[np.newaxis, np.newaxis, :]
        if settings.enable_eke_diss_bottom:
            forc = update(forc, at[2:-2, 2:-2, :], np.where(mask, a_loc[2:-2, 2:-2, np.newaxis] /
                                                  vs.dzw[np.newaxis, np.newaxis, :], forc[2:-2, 2:-2, :]))
        else:
            forc = update(forc, at[2:-2, 2:-2, :], np.where(mask, settings.eke_diss_surfbot_frac * a_loc[2:-2, 2:-2, np.newaxis]
                                           / vs.dzw[np.newaxis, np.newaxis, :], forc[2:-2, 2:-2, :]))
            forc = update(forc, at[2:-2, 2:-2, -1], (1. - settings.eke_diss_surfbot_frac) \
                                    * a_loc[2:-2, 2:-2] / (0.5 * vs.dzw[-1]))

    """
    forcing by bottom friction
    """
    if not settings.enable_store_bottom_friction_tke:
        forc = forc + vs.K_diss_bot

    """
    prevent negative dissipation of IW energy
    """
    maxE_iw = np.maximum(0., vs.E_iw[:, :, :, vs.tau])

    """
    vertical diffusion and dissipation is solved implicitly
    """
    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    delta = update(delta, at[:, :, :-1], settings.dt_tracer * settings.tau_v / vs.dzt[np.newaxis, np.newaxis, 1:] * 0.5 \
        * (vs.c0[2:-2, 2:-2, :-1] + vs.c0[2:-2, 2:-2, 1:]))
    delta = update(delta, at[:, :, -1], 0.)
    a_tri = update(a_tri, at[:, :, 1:-1], -delta[:, :, :-2] * vs.c0[2:-2, 2:-2, :-2] \
        / vs.dzw[np.newaxis, np.newaxis, 1:-1])
    a_tri = update(a_tri, at[:, :, -1], -delta[:, :, -2] / (0.5 * vs.dzw[-1:]) * vs.c0[2:-2, 2:-2, -2])
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + delta[:, :, 1:-1] * vs.c0[2:-2, 2:-2, 1:-1] / vs.dzw[np.newaxis, np.newaxis, 1:-1] \
        + delta[:, :, :-2] * vs.c0[2:-2, 2:-2, 1:-1] / vs.dzw[np.newaxis, np.newaxis, 1:-1] \
        + settings.dt_tracer * vs.alpha_c[2:-2, 2:-2, 1:-1] * maxE_iw[2:-2, 2:-2, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / (0.5 * vs.dzw[-1:]) * vs.c0[2:-2, 2:-2, -1] \
        + settings.dt_tracer * vs.alpha_c[2:-2, 2:-2, -1] * maxE_iw[2:-2, 2:-2, -1])
    b_tri_edge = 1 + delta / vs.dzw * vs.c0[2:-2, 2:-2, :] \
        + settings.dt_tracer * vs.alpha_c[2:-2, 2:-2, :] * maxE_iw[2:-2, 2:-2, :]
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / \
        vs.dzw[np.newaxis, np.newaxis, :-1] * vs.c0[2:-2, 2:-2, 1:])
    d_tri = update(d_tri, at[...], vs.E_iw[2:-2, 2:-2, :, vs.tau] + settings.dt_tracer * forc[2:-2, 2:-2, :])
    d_tri_edge = d_tri + settings.dt_tracer * \
        vs.forc_iw_bottom[2:-2, 2:-2, np.newaxis] / vs.dzw[np.newaxis, np.newaxis, :]
    d_tri = update_add(d_tri, at[:, :, -1], settings.dt_tracer * vs.forc_iw_surface[2:-2, 2:-2] / (0.5 * vs.dzw[-1:]))

    sol = utilities.solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, d_edge=d_tri_edge, edge_mask=edge_mask)
    vs.E_iw = update(vs.E_iw, at[2:-2, 2:-2, :, vs.taup1], np.where(water_mask, sol, vs.E_iw[2:-2, 2:-2, :, vs.taup1]))

    """
    store IW dissipation
    """
    vs.iw_diss = vs.alpha_c * maxE_iw * vs.E_iw[..., vs.taup1]

    """
    add tendency due to lateral diffusion
    """
    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    if settings.enable_idemix_hor_diffusion:
        flux_east = update(flux_east, at[:-1, :, :], settings.tau_h * 0.5 * (vs.v0[1:, :, :] + vs.v0[:-1, :, :]) \
            * (vs.v0[1:, :, :] * vs.E_iw[1:, :, :, vs.tau] - vs.v0[:-1, :, :] * vs.E_iw[:-1, :, :, vs.tau]) \
            / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) * vs.maskU[:-1, :, :])
        if runtime_settings.pyom_compatibility_mode:
            flux_east = update(flux_east, at[-5, :, :], 0.)
        else:
            flux_east = update(flux_east, at[-1, :, :], 0.)
        flux_north = update(flux_north, at[:, :-1, :], settings.tau_h * 0.5 * (vs.v0[:, 1:, :] + vs.v0[:, :-1, :]) \
            * (vs.v0[:, 1:, :] * vs.E_iw[:, 1:, :, vs.tau] - vs.v0[:, :-1, :] * vs.E_iw[:, :-1, :, vs.tau]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis])
        flux_north = update(flux_north, at[:, -1, :], 0.)
        vs.E_iw = update_add(vs.E_iw, at[2:-2, 2:-2, :, vs.taup1], settings.dt_tracer * vs.maskW[2:-2, 2:-2, :] \
            * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
               / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
               + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
               / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))

    """
    add tendency due to advection
    """
    if settings.enable_idemix_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(state, vs.E_iw[:, :, :, vs.tau])

    if settings.enable_idemix_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(state, vs.E_iw[:, :, :, vs.tau])

    if settings.enable_idemix_superbee_advection or settings.enable_idemix_upwind_advection:
        vs.dE_iw = update(vs.dE_iw, at[2:-2, 2:-2, :, vs.tau], vs.maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                            / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                            - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                            / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))
        vs.dE_iw = update_add(vs.dE_iw, at[:, :, 0, vs.tau], -flux_top[:, :, 0] / vs.dzw[0:1])
        vs.dE_iw = update_add(vs.dE_iw, at[:, :, 1:-1, vs.tau], -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) \
            / vs.dzw[np.newaxis, np.newaxis, 1:-1])
        vs.dE_iw = update_add(vs.dE_iw, at[:, :, -1, vs.tau], - \
            (flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * vs.dzw[-1:]))

        """
        Adam Bashforth time stepping
        """
        vs.E_iw = update_add(vs.E_iw, at[:, :, :, vs.taup1], settings.dt_tracer * ((1.5 + settings.AB_eps) * vs.dE_iw[:, :, :, vs.tau]
                                             - (0.5 + settings.AB_eps) * vs.dE_iw[:, :, :, vs.taum1]))

    return KernelOutput(E_iw=vs.E_iw, dE_iw=vs.dE_iw, iw_diss=vs.iw_diss)


@veros_kernel
def gofx2(x, pi):
    x = np.maximum(3., x)
    c = 1. - (2. / pi) * np.arcsin(1. / x)
    return 2. / pi / c * 0.9 * x**(-2. / 3.) * (1 - np.exp(-x / 4.3))


@veros_kernel
def hofx1(x, pi):
    eps = np.finfo(x.dtype).eps  # prevent division by zero
    x = np.maximum(1. + eps, x)
    return (2. / pi) / (1. - (2. / pi) * np.arcsin(1. / x)) * (x - 1.) / (x + 1.)


@veros_routine
def integrate_idemix(state):
    vs = state.variables
    vs.update(integrate_idemix_kernel(state))
