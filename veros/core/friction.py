from veros.core.operators import numpy as np

from veros import veros_routine, veros_kernel, KernelOutput
from veros.variables import allocate
from veros.core import numerics, utilities, isoneutral
from veros.core.operators import update, update_add, at


@veros_kernel
def explicit_vert_friction(state):
    """
    explicit vertical friction
    dissipation is calculated and added to K_diss_v
    """
    vs = state.variables

    diss = allocate(state.dimensions, ("xt", "yu", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yu", "zt"))

    """
    vertical friction of zonal momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[2:-1, 1:-2, :-1])
    flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
        / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    flux_top = update(flux_top, at[:, :, -1], 0.0)
    vs.du_mix = update(vs.du_mix, at[:, :, 0], flux_top[:, :, 0] / vs.dzt[0] * vs.maskU[:, :, 0])
    vs.du_mix = update(vs.du_mix, at[:, :, 1:], (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / vs.dzt[1:] * vs.maskU[:, :, 1:])

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    diss = update(diss, at[1:-2, 1:-2, :-1], (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
        * flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
    diss = update(diss, at[:, :, -1], 0.0)
    diss = numerics.ugrid_to_tgrid(state, diss)
    vs.K_diss_v = vs.K_diss_v + diss

    """
    vertical friction of meridional momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[1:-2, 2:-1, :-1])
    flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
        / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskV[1:-2, 1:-2, 1:] \
        * vs.maskV[1:-2, 1:-2, :-1])
    flux_top = update(flux_top, at[:, :, -1], 0.0)
    vs.dv_mix = update(vs.dv_mix, at[:, :, 1:], (flux_top[:, :, 1:] - flux_top[:, :, :-1]) \
        / vs.dzt[np.newaxis, np.newaxis, 1:] * vs.maskV[:, :, 1:])
    vs.dv_mix = update(vs.dv_mix, at[:, :, 0], flux_top[:, :, 0] / vs.dzt[0] * vs.maskV[:, :, 0])

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    diss = update(diss, at[1:-2, 1:-2, :-1], (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
        * flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
    diss = update(diss, at[:, :, -1], 0.0)
    diss = numerics.vgrid_to_tgrid(state, diss)
    vs.K_diss_v = vs.K_diss_v + diss

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_v=vs.K_diss_v)


@veros_kernel
def implicit_vert_friction(state):
    """
    vertical friction
    dissipation is calculated and added to K_diss_v
    """
    vs = state.variables
    settings = state.settings

    diss = allocate(state.dimensions, ("xt", "yu", "zt"))
    a_tri = allocate(state.dimensions, ("xt", "yu", "zt"))[1:-2, 1:-2]
    b_tri = allocate(state.dimensions, ("xt", "yu", "zt"))[1:-2, 1:-2]
    c_tri = allocate(state.dimensions, ("xt", "yu", "zt"))[1:-2, 1:-2]
    d_tri = allocate(state.dimensions, ("xt", "yu", "zt"))[1:-2, 1:-2]
    delta = allocate(state.dimensions, ("xt", "yu", "zt"))[1:-2, 1:-2]
    flux_top = allocate(state.dimensions, ("xt", "yu", "zt"))

    """
    implicit vertical friction of zonal momentum
    """
    kss = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[2:-1, 1:-2])
    _, water_mask, edge_mask = utilities.create_water_masks(kss, settings.nz)

    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[2:-1, 1:-2, :-1])
    delta = update(delta, at[:, :, :-1], settings.dt_mom / vs.dzw[:-1] * fxa * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update(b_tri, at[:, :, 1:], 1 + delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update_add(b_tri, at[:, :, 1:-1], delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    c_tri = update(c_tri, at[...], -delta / vs.dzt[np.newaxis, np.newaxis, :])
    d_tri = update(d_tri, at[...], vs.u[1:-2, 1:-2, :, vs.tau])

    res = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    vs.u = update(vs.u, at[1:-2, 1:-2, :, vs.taup1], np.where(water_mask, res, vs.u[1:-2, 1:-2, :, vs.taup1]))
    vs.du_mix = update(vs.du_mix, at[1:-2, 1:-2], (vs.u[1:-2, 1:-2, :, vs.taup1] - vs.u[1:-2, 1:-2, :, vs.tau]) / settings.dt_mom)

    """
    diagnose dissipation by vertical friction of zonal momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[2:-1, 1:-2, :-1])
    flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (vs.u[1:-2, 1:-2, 1:, vs.taup1] - vs.u[1:-2, 1:-2, :-1, vs.taup1]) \
        / vs.dzw[:-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    diss = update(diss, at[1:-2, 1:-2, :-1], (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
        * flux_top[1:-2, 1:-2, :-1] / vs.dzw[:-1])
    diss = update(diss, at[:, :, -1], 0.0)
    diss = numerics.ugrid_to_tgrid(state, diss)
    vs.K_diss_v = vs.K_diss_v + diss

    """
    implicit vertical friction of meridional momentum
    """
    kss = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[1:-2, 2:-1])
    _, water_mask, edge_mask = utilities.create_water_masks(kss, settings.nz)

    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[1:-2, 2:-1, :-1])
    delta = update(delta, at[:, :, :-1], settings.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update(b_tri, at[:, :, 1:], 1 + delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update_add(b_tri, at[:, :, 1:-1], delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1])
    c_tri = update(c_tri, at[:, :, -1], 0.)
    d_tri = update(d_tri, at[...], vs.v[1:-2, 1:-2, :, vs.tau])

    res = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    vs.v = update(vs.v, at[1:-2, 1:-2, :, vs.taup1], np.where(water_mask, res, vs.v[1:-2, 1:-2, :, vs.taup1]))
    vs.dv_mix = update(vs.dv_mix, at[1:-2, 1:-2], (vs.v[1:-2, 1:-2, :, vs.taup1] - vs.v[1:-2, 1:-2, :, vs.tau]) / settings.dt_mom)

    """
    diagnose dissipation by vertical friction of meridional momentum
    """
    fxa = 0.5 * (vs.kappaM[1:-2, 1:-2, :-1] + vs.kappaM[1:-2, 2:-1, :-1])
    flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (vs.v[1:-2, 1:-2, 1:, vs.taup1] - vs.v[1:-2, 1:-2, :-1, vs.taup1]) \
        / vs.dzw[:-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
    diss = update(diss, at[1:-2, 1:-2, :-1], (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
        * flux_top[1:-2, 1:-2, :-1] / vs.dzw[:-1])
    diss = update(diss, at[:, :, -1], 0.0)
    diss = numerics.vgrid_to_tgrid(state, diss)
    vs.K_diss_v = vs.K_diss_v + diss

    return KernelOutput(u=vs.u, v=vs.v, du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_v=vs.K_diss_v)


@veros_kernel
def rayleigh_friction(state):
    """
    interior Rayleigh friction
    dissipation is calculated and added to K_diss_bot
    """
    vs = state.variables
    settings = state.settings

    vs.du_mix = update_add(vs.du_mix, at[...], -1 * vs.maskU * settings.r_ray * vs.u[..., vs.tau])
    if settings.enable_conserve_energy:
        diss = vs.maskU * settings.r_ray * vs.u[..., vs.tau]**2
        vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_u(state, diss))

    vs.dv_mix = update_add(vs.dv_mix, at[...], -1 * vs.maskV * settings.r_ray * vs.v[..., vs.tau])
    if settings.enable_conserve_energy:
        diss = vs.maskV * settings.r_ray * vs.v[..., vs.tau]**2
        vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_v(state, diss))

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_bot=vs.K_diss_bot)


@veros_kernel
def linear_bottom_friction(state):
    """
    linear bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_bottom_friction_var:
        """
        with spatially varying coefficient
        """
        k = np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
        mask = np.arange(settings.nz) == k[:, :, np.newaxis]
        vs.du_mix = update_add(vs.du_mix, at[1:-2, 2:-2], -(vs.maskU[1:-2, 2:-2] * vs.r_bot_var_u[1:-2, 2:-2, np.newaxis]) \
            * vs.u[1:-2, 2:-2, :, vs.tau] * mask)
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(diss, at[1:-2, 2:-2], vs.maskU[1:-2, 2:-2] * vs.r_bot_var_u[1:-2, 2:-2, np.newaxis] \
                * vs.u[1:-2, 2:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_u(state, diss))

        k = np.maximum(vs.kbot[2:-2, 2:-1], vs.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(settings.nz) == k[:, :, np.newaxis]
        vs.dv_mix = update_add(vs.dv_mix, at[2:-2, 1:-2], -(vs.maskV[2:-2, 1:-2] * vs.r_bot_var_v[2:-2, 1:-2, np.newaxis]) \
            * vs.v[2:-2, 1:-2, :, vs.tau] * mask)
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(diss, at[2:-2, 1:-2], vs.maskV[2:-2, 1:-2] * vs.r_bot_var_v[2:-2, 1:-2, np.newaxis] \
                * vs.v[2:-2, 1:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_v(state, diss))
    else:
        """
        with constant coefficient
        """
        k = np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
        mask = np.arange(settings.nz) == k[:, :, np.newaxis]

        vs.du_mix = update_add(vs.du_mix, at[1:-2, 2:-2], -1 * vs.maskU[1:-2, 2:-2] * settings.r_bot * vs.u[1:-2, 2:-2, :, vs.tau] * mask)
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(diss, at[1:-2, 2:-2], vs.maskU[1:-2, 2:-2] * settings.r_bot * vs.u[1:-2, 2:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_u(state, diss))

        k = np.maximum(vs.kbot[2:-2, 2:-1], vs.kbot[2:-2, 1:-2]) - 1
        mask = np.arange(settings.nz) == k[:, :, np.newaxis]

        vs.dv_mix = update_add(vs.dv_mix, at[2:-2, 1:-2], -1 * vs.maskV[2:-2, 1:-2] * settings.r_bot * vs.v[2:-2, 1:-2, :, vs.tau] * mask)
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(diss, at[2:-2, 1:-2], vs.maskV[2:-2, 1:-2] * settings.r_bot * vs.v[2:-2, 1:-2, :, vs.tau]**2 * mask)
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_v(state, diss))

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_bot=vs.K_diss_bot)


@veros_kernel
def quadratic_bottom_friction(state):
    """
    quadratic bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    vs = state.variables
    settings = state.settings

    # we might want to account for EKE in the drag, also a tidal residual
    k = np.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
    mask = k[..., np.newaxis] == np.arange(settings.nz)[np.newaxis, np.newaxis, :]
    fxa = vs.maskV[1:-2, 2:-2, :] * vs.v[1:-2, 2:-2, :, vs.tau]**2 \
        + vs.maskV[1:-2, 1:-3, :] * vs.v[1:-2, 1:-3, :, vs.tau]**2 \
        + vs.maskV[2:-1, 2:-2, :] * vs.v[2:-1, 2:-2, :, vs.tau]**2 \
        + vs.maskV[2:-1, 1:-3, :] * vs.v[2:-1, 1:-3, :, vs.tau]**2
    fxa = np.sqrt(vs.u[1:-2, 2:-2, :, vs.tau]**2 + 0.25 * fxa)
    aloc = vs.maskU[1:-2, 2:-2, :] * settings.r_quad_bot * vs.u[1:-2, 2:-2, :, vs.tau] \
        * fxa / vs.dzt[np.newaxis, np.newaxis, :] * mask
    vs,du_mix = update_add(vs.du_mix, at[1:-2, 2:-2, :], -aloc)

    if settings.enable_conserve_energy:
        diss = allocate(state.dimensions, ("xt", "yu", "zt"))
        diss = update(diss, at[1:-2, 2:-2, :], aloc * vs.u[1:-2, 2:-2, :, vs.tau])
        vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_u(state, diss))

    k = np.maximum(vs.kbot[2:-2, 1:-2], vs.kbot[2:-2, 2:-1]) - 1
    mask = k[..., np.newaxis] == np.arange(settings.nz)[np.newaxis, np.newaxis, :]
    fxa = vs.maskU[2:-2, 1:-2, :] * vs.u[2:-2, 1:-2, :, vs.tau]**2 \
        + vs.maskU[1:-3, 1:-2, :] * vs.u[1:-3, 1:-2, :, vs.tau]**2 \
        + vs.maskU[2:-2, 2:-1, :] * vs.u[2:-2, 2:-1, :, vs.tau]**2 \
        + vs.maskU[1:-3, 2:-1, :] * vs.u[1:-3, 2:-1, :, vs.tau]**2
    fxa = np.sqrt(vs.v[2:-2, 1:-2, :, vs.tau]**2 + 0.25 * fxa)
    aloc = vs.maskV[2:-2, 1:-2, :] * settings.r_quad_bot * vs.v[2:-2, 1:-2, :, vs.tau] \
        * fxa / vs.dzt[np.newaxis, np.newaxis, :] * mask
    vs.dv_mix = update_add(vs.dv_mix, at[2:-2, 1:-2, :], -aloc)

    if settings.enable_conserve_energy:
        diss = allocate(state.dimensions, ("xt", "yu", "zt"))
        diss = update(diss, at[2:-2, 1:-2, :], aloc * vs.v[2:-2, 1:-2, :, vs.tau])
        vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_v(state, diss))

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_bot=vs.K_diss_bot)


@veros_kernel
def harmonic_friction(state):
    """
    horizontal harmonic friction
    dissipation is calculated and added to K_diss_h
    """
    vs = state.variables
    settings = state.settings

    diss = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_east = allocate(state.dimensions, ("xu", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yu", "zt"))

    """
    Zonal velocity
    """
    if settings.enable_hor_friction_cos_scaling:
        fxa = vs.cost**settings.hor_friction_cosPower
        flux_east = update(flux_east, at[:-1], settings.A_h * fxa[np.newaxis, :, np.newaxis] * (vs.u[1:, :, :, vs.tau] - vs.u[:-1, :, :, vs.tau]) \
            / (vs.cost * vs.dxt[1:, np.newaxis])[:, :, np.newaxis] * vs.maskU[1:] * vs.maskU[:-1])
        fxa = vs.cosu**settings.hor_friction_cosPower
        flux_north = update(flux_north, at[:, :-1], settings.A_h * fxa[np.newaxis, :-1, np.newaxis] * (vs.u[:, 1:, :, vs.tau] - vs.u[:, :-1, :, vs.tau]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:] * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])
        if settings.enable_noslip_lateral:
            flux_north = update_add(flux_north, at[:, :-1], 2 * settings.A_h * fxa[np.newaxis, :-1, np.newaxis] * (vs.u[:, 1:, :, vs.tau]) \
                / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:] * (1 - vs.maskU[:, :-1]) * vs.cosu[np.newaxis, :-1, np.newaxis]\
                - 2 * settings.A_h * fxa[np.newaxis, :-1, np.newaxis] * (vs.u[:, :-1, :, vs.tau]) \
                / vs.dyu[np.newaxis, :-1, np.newaxis] * (1 - vs.maskU[:, 1:]) * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])
    else:
        flux_east = update(flux_east, at[:-1, :, :], settings.A_h * (vs.u[1:, :, :, vs.tau] - vs.u[:-1, :, :, vs.tau]) \
            / (vs.cost * vs.dxt[1:, np.newaxis])[:, :, np.newaxis] * vs.maskU[1:] * vs.maskU[:-1])
        flux_north = update(flux_north, at[:, :-1, :], settings.A_h * (vs.u[:, 1:, :, vs.tau] - vs.u[:, :-1, :, vs.tau]) \
            / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:] * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])
        if settings.enable_noslip_lateral:
            flux_north = update_add(flux_north, at[:, :-1], 2 * settings.A_h * vs.u[:, 1:, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
                * vs.maskU[:, 1:] * (1 - vs.maskU[:, :-1]) * vs.cosu[np.newaxis, :-1, np.newaxis]\
                - 2 * settings.A_h * vs.u[:, :-1, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
                * (1 - vs.maskU[:, 1:]) * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])

    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, -1, :], 0.)

    """
    update tendency
    """
    vs.du_mix = update_add(vs.du_mix, at[2:-2, 2:-2, :], vs.maskU[2:-2, 2:-2] * ((flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2])
                                                  / (vs.cost[2:-2] * vs.dxu[2:-2, np.newaxis])[:, :, np.newaxis]
                                                  + (flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3])
                                                  / (vs.cost[2:-2] * vs.dyt[2:-2])[np.newaxis, :, np.newaxis]))

    if settings.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss = update(diss, at[1:-2, 2:-2], 0.5 * ((vs.u[2:-1, 2:-2, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * flux_east[1:-2, 2:-2]
                                  + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[:-3, 2:-2, :, vs.tau]) * flux_east[:-3, 2:-2]) \
            / (vs.cost[2:-2] * vs.dxu[1:-2, np.newaxis])[:, :, np.newaxis]\
            + 0.5 * ((vs.u[1:-2, 3:-1, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * flux_north[1:-2, 2:-2]
                     + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[1:-2, 1:-3, :, vs.tau]) * flux_north[1:-2, 1:-3]) \
            / (vs.cost[2:-2] * vs.dyt[2:-2])[np.newaxis, :, np.newaxis])
        vs.K_diss_h = numerics.calc_diss_u(state, diss)

    """
    Meridional velocity
    """
    if settings.enable_hor_friction_cos_scaling:
        flux_east = update(flux_east, at[:-1], settings.A_h * vs.cosu[np.newaxis, :, np.newaxis] ** settings.hor_friction_cosPower \
            * (vs.v[1:, :, :, vs.tau] - vs.v[:-1, :, :, vs.tau]) \
            / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * vs.maskV[1:] * vs.maskV[:-1])

        if settings.enable_noslip_lateral:
            flux_east = update_add(flux_east, at[:-1], 2 * settings.A_h * fxa[np.newaxis, :, np.newaxis] * vs.v[1:, :, :, vs.tau] \
                / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * vs.maskV[1:] * (1 - vs.maskV[:-1]) \
                - 2 * settings.A_h * fxa[np.newaxis, :, np.newaxis] * vs.v[:-1, :, :, vs.tau] \
                / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * (1 - vs.maskV[1:]) * vs.maskV[:-1])

        flux_north = update(flux_north, at[:, :-1], settings.A_h * vs.cost[np.newaxis, 1:, np.newaxis] ** settings.hor_friction_cosPower \
            * (vs.v[:, 1:, :, vs.tau] - vs.v[:, :-1, :, vs.tau]) \
            / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] * vs.maskV[:, :-1] * vs.maskV[:, 1:])
    else:
        flux_east = update(flux_east, at[:-1], settings.A_h * (vs.v[1:, :, :, vs.tau] - vs.v[:-1, :, :, vs.tau]) \
            / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] * vs.maskV[1:] * vs.maskV[:-1])

        if settings.enable_noslip_lateral:
            flux_east = update_add(flux_east, at[:-1], 2 * settings.A_h * vs.v[1:, :, :, vs.tau] / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] \
                * vs.maskV[1:] * (1 - vs.maskV[:-1]) \
                - 2 * settings.A_h * vs.v[:-1, :, :, vs.tau] / (vs.cosu * vs.dxu[:-1, np.newaxis])[:, :, np.newaxis] \
                * (1 - vs.maskV[1:]) * vs.maskV[:-1])

        flux_north = update(flux_north, at[:, :-1], settings.A_h * (vs.v[:, 1:, :, vs.tau] - vs.v[:, :-1, :, vs.tau]) \
            / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] * vs.maskV[:, :-1] * vs.maskV[:, 1:])

    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, -1, :], 0.)

    """
    update tendency
    """
    vs.dv_mix = update_add(vs.dv_mix, at[2:-2, 2:-2], vs.maskV[2:-2, 2:-2] * ((flux_east[2:-2, 2:-2] - flux_east[1:-3, 2:-2])
                                               / (vs.cosu[2:-2] * vs.dxt[2:-2, np.newaxis])[:, :, np.newaxis]
                                               + (flux_north[2:-2, 2:-2] - flux_north[2:-2, 1:-3])
                                               / (vs.dyu[2:-2] * vs.cosu[2:-2])[np.newaxis, :, np.newaxis]))

    if settings.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        diss = update(diss, at[2:-2, 1:-2], 0.5 * ((vs.v[3:-1, 1:-2, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * flux_east[2:-2, 1:-2]
                                  + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[1:-3, 1:-2, :, vs.tau]) * flux_east[1:-3, 1:-2]) \
            / (vs.cosu[1:-2] * vs.dxt[2:-2, np.newaxis])[:, :, np.newaxis] \
            + 0.5 * ((vs.v[2:-2, 2:-1, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * flux_north[2:-2, 1:-2]
                     + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[2:-2, :-3, :, vs.tau]) * flux_north[2:-2, :-3]) \
            / (vs.cosu[1:-2] * vs.dyu[1:-2])[np.newaxis, :, np.newaxis])
        vs.K_diss_h = update_add(vs.K_diss_h, at[...], numerics.calc_diss_v(state, diss))

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_h=vs.K_diss_h)


@veros_kernel
def biharmonic_friction(state):
    """
    horizontal biharmonic friction
    dissipation is calculated and added to K_diss_h
    """
    vs = state.variables
    settings = state.settings

    flux_east = allocate(state.dimensions, ("xu", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yu", "zt"))
    fxa = np.sqrt(abs(settings.A_hbi))

    """
    Zonal velocity
    """
    flux_east = update(flux_east, at[:-1, :, :], fxa * (vs.u[1:, :, :, vs.tau] - vs.u[:-1, :, :, vs.tau]) \
        / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
        * vs.maskU[1:, :, :] * vs.maskU[:-1, :, :])
    flux_north = update(flux_north, at[:, :-1, :], fxa * (vs.u[:, 1:, :, vs.tau] - vs.u[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:, :] \
        * vs.maskU[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis])
    if settings.enable_noslip_lateral:
        flux_north = update_add(flux_north, at[:, :-1], 2 * fxa * vs.u[:, 1:, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * vs.maskU[:, 1:] * (1 - vs.maskU[:, :-1]) * vs.cosu[np.newaxis, :-1, np.newaxis]\
            - 2 * fxa * vs.u[:, :-1, :, vs.tau] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * (1 - vs.maskU[:, 1:]) * vs.maskU[:, :-1] * vs.cosu[np.newaxis, :-1, np.newaxis])
    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, -1, :], 0.)

    del2 = allocate(state.dimensions, ("xt", "yu", "zt"))
    del2 = update(del2, at[1:, 1:, :], (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :]) \
        / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dxu[1:, np.newaxis, np.newaxis]) \
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :]) \
        / (vs.cost[np.newaxis, 1:, np.newaxis] * vs.dyt[np.newaxis, 1:, np.newaxis]))

    flux_east = update(flux_east, at[:-1, :, :], fxa * (del2[1:, :, :] - del2[:-1, :, :]) \
        / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis]) \
        * vs.maskU[1:, :, :] * vs.maskU[:-1, :, :])
    flux_north = update(flux_north, at[:, :-1, :], fxa * (del2[:, 1:, :] - del2[:, :-1, :]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskU[:, 1:, :] \
        * vs.maskU[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis])
    if settings.enable_noslip_lateral:
        flux_north = update_add(flux_north, at[:, :-1, :], 2 * fxa * del2[:, 1:, :] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * vs.maskU[:, 1:, :] * (1 - vs.maskU[:, :-1, :]) * vs.cosu[np.newaxis, :-1, np.newaxis] \
            - 2 * fxa * del2[:, :-1, :] / vs.dyu[np.newaxis, :-1, np.newaxis] \
            * (1 - vs.maskU[:, 1:, :]) * vs.maskU[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis])
    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, -1, :], 0.)

    """
    update tendency
    """
    vs.du_mix = update_add(vs.du_mix, at[2:-2, 2:-2, :], -1 * vs.maskU[2:-2, 2:-2, :] * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                      / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[2:-2, np.newaxis, np.newaxis])
                                                      + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                      / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))
    if settings.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        flux_east = utilities.enforce_boundaries(flux_east, settings.enable_cyclic_x)
        flux_north = utilities.enforce_boundaries(flux_north, settings.enable_cyclic_x)
        diss = allocate(state.dimensions, ("xt", "yu", "zt"))
        diss = update(diss, at[1:-2, 2:-2, :], -0.5 * ((vs.u[2:-1, 2:-2, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * flux_east[1:-2, 2:-2, :]
                                      + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[:-3, 2:-2, :, vs.tau]) * flux_east[:-3, 2:-2, :]) \
            / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[1:-2, np.newaxis, np.newaxis])  \
            - 0.5 * ((vs.u[1:-2, 3:-1, :, vs.tau] - vs.u[1:-2, 2:-2, :, vs.tau]) * flux_north[1:-2, 2:-2, :]
                     + (vs.u[1:-2, 2:-2, :, vs.tau] - vs.u[1:-2, 1:-3, :, vs.tau]) * flux_north[1:-2, 1:-3, :]) \
            / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis]))
        vs.K_diss_h = numerics.calc_diss_u(state, diss)

    """
    Meridional velocity
    """
    flux_east = update(flux_east, at[:-1, :, :], fxa * (vs.v[1:, :, :, vs.tau] - vs.v[:-1, :, :, vs.tau]) \
        / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
        * vs.maskV[1:, :, :] * vs.maskV[:-1, :, :])
    if settings.enable_noslip_lateral:
        flux_east = update_add(flux_east, at[:-1, :, :], 2 * fxa * vs.v[1:, :, :, vs.tau] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * vs.maskV[1:, :, :] * (1 - vs.maskV[:-1, :, :]) \
            - 2 * fxa * vs.v[:-1, :, :, vs.tau] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * (1 - vs.maskV[1:, :, :]) * vs.maskV[:-1, :, :])
    flux_north = update(flux_north, at[:, :-1, :], fxa * (vs.v[:, 1:, :, vs.tau] - vs.v[:, :-1, :, vs.tau]) \
        / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] \
        * vs.maskV[:, :-1, :] * vs.maskV[:, 1:, :])
    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, -1, :], 0.)

    del2 = update(del2, at[1:, 1:, :], (flux_east[1:, 1:, :] - flux_east[:-1, 1:, :]) \
        / (vs.cosu[np.newaxis, 1:, np.newaxis] * vs.dxt[1:, np.newaxis, np.newaxis])  \
        + (flux_north[1:, 1:, :] - flux_north[1:, :-1, :]) \
        / (vs.dyu[np.newaxis, 1:, np.newaxis] * vs.cosu[np.newaxis, 1:, np.newaxis]))

    flux_east = update(flux_east, at[:-1, :, :], fxa * (del2[1:, :, :] - del2[:-1, :, :]) \
        / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
        * vs.maskV[1:, :, :] * vs.maskV[:-1, :, :])
    if settings.enable_noslip_lateral:
        flux_east = update_add(flux_east, at[:-1, :, :], 2 * fxa * del2[1:, :, :] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * vs.maskV[1:, :, :] * (1 - vs.maskV[:-1, :, :]) \
            - 2 * fxa * del2[:-1, :, :] / (vs.cosu[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) \
            * (1 - vs.maskV[1:, :, :]) * vs.maskV[:-1, :, :])
    flux_north = update(flux_north, at[:, :-1, :], fxa * (del2[:, 1:, :] - del2[:, :-1, :]) \
        / vs.dyt[np.newaxis, 1:, np.newaxis] * vs.cost[np.newaxis, 1:, np.newaxis] \
        * vs.maskV[:, :-1, :] * vs.maskV[:, 1:, :])
    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, -1, :], 0.)

    """
    update tendency
    """
    vs.dv_mix = update_add(vs.dv_mix, at[2:-2, 2:-2, :], -1 * vs.maskV[2:-2, 2:-2, :] * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                      / (vs.cosu[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                      + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                      / (vs.dyu[np.newaxis, 2:-2, np.newaxis] * vs.cosu[np.newaxis, 2:-2, np.newaxis])))

    if settings.enable_conserve_energy:
        """
        diagnose dissipation by lateral friction
        """
        flux_east = utilities.enforce_boundaries(flux_east, settings.enable_cyclic_x)
        flux_north = utilities.enforce_boundaries(flux_north, settings.enable_cyclic_x)
        diss = update(diss, at[2:-2, 1:-2, :], -0.5 * ((vs.v[3:-1, 1:-2, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * flux_east[2:-2, 1:-2, :]
                                      + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[1:-3, 1:-2, :, vs.tau]) * flux_east[1:-3, 1:-2, :]) \
            / (vs.cosu[np.newaxis, 1:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis]) \
            - 0.5 * ((vs.v[2:-2, 2:-1, :, vs.tau] - vs.v[2:-2, 1:-2, :, vs.tau]) * flux_north[2:-2, 1:-2, :]
                     + (vs.v[2:-2, 1:-2, :, vs.tau] - vs.v[2:-2, :-3, :, vs.tau]) * flux_north[2:-2, :-3, :]) \
            / (vs.cosu[np.newaxis, 1:-2, np.newaxis] * vs.dyu[np.newaxis, 1:-2, np.newaxis]))
        vs.K_diss_h = update_add(vs.K_diss_h, at[...], numerics.calc_diss_v(state, diss))

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_h=vs.K_diss_h)


@veros_kernel
def momentum_sources(state):
    """
    other momentum sources
    dissipation is calculated and added to K_diss_bot
    """
    vs = state.variables
    settings = state.settings

    vs.du_mix = update_add(vs.du_mix, at[...], vs.maskU * vs.u_source)
    if settings.enable_conserve_energy:
        diss = -1 * vs.maskU * vs.u[..., vs.tau] * vs.u_source
        vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_u(state, diss))

    vs.dv_mix = update_add(vs.dv_mix, at[...], vs.maskV * vs.v_source)
    if settings.enable_conserve_energy:
        diss = -1 * vs.maskV * vs.v[..., vs.tau] * vs.v_source
        vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_v(state, diss))

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_bot=vs.K_diss_bot)


@veros_routine
def friction(state):
    """
    vertical friction
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_implicit_vert_friction:
        vs.update(implicit_vert_friction(state))

    if settings.enable_explicit_vert_friction:
        vs.update(explicit_vert_friction(state))

    """
    TEM formalism for eddy-driven velocity
    """
    if settings.enable_TEM_friction:
        vs.update(isoneutral.isoneutral_friction(state))

    """
    horizontal friction
    """
    if settings.enable_hor_friction:
        vs.update(harmonic_friction(state))

    if settings.enable_biharmonic_friction:
        vs.update(biharmonic_friction(state))

    """
    Rayleigh and bottom friction
    """
    if settings.enable_ray_friction:
        vs.update(rayleigh_friction(state))

    if settings.enable_bottom_friction:
        vs.update(linear_bottom_friction(state))

    if settings.enable_quadratic_bottom_friction:
        vs.update(quadratic_bottom_friction(state))

    """
    add user defined forcing
    """
    if settings.enable_momentum_sources:
        vs.update(momentum_sources(state))
