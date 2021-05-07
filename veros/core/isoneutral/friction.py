from veros.core.operators import numpy as np

from veros import veros_kernel, KernelOutput
from veros.variables import allocate
from veros.core import numerics, utilities
from veros.core.operators import update, update_add, at


@veros_kernel
def isoneutral_friction(state):
    """
    vertical friction using TEM formalism for eddy driven velocity
    """
    vs = state.variables
    settings = state.settings

    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    if settings.enable_implicit_vert_friction:
        aloc = vs.u[:, :, :, vs.taup1]
    else:
        aloc = vs.u[:, :, :, vs.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[2:-1, 1:-2])
    _, water_mask, edge_mask = utilities.create_water_masks(ks, settings.nz)

    fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :] + vs.kappa_gm[2:-1, 1:-2, :])

    delta, a_tri, b_tri, c_tri = (
        allocate(state.dimensions, ("xt", "yt", "zt"))[1:-2, 1:-2, :]
        for _ in range(4)
    )

    delta = update(delta, at[:, :, :-1], settings.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
    delta = update(delta, at[-1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / vs.dzt[-1])
    c_tri = update(c_tri, at[...], - delta / vs.dzt[np.newaxis, np.newaxis, :])

    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    vs.u = update(vs.u, at[1:-2, 1:-2, :, vs.taup1], np.where(water_mask, sol, vs.u[1:-2, 1:-2, :, vs.taup1]))
    vs.du_mix = update_add(vs.du_mix, at[1:-2, 1:-2, :], (vs.u[1:-2, 1:-2, :, vs.taup1] - aloc[1:-2, 1:-2, :]) / settings.dt_mom * water_mask)

    if settings.enable_conserve_energy:
        # diagnose dissipation
        diss = np.zeros_like(vs.maskU)
        fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :-1] + vs.kappa_gm[2:-1, 1:-2, :-1])
        flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (vs.u[1:-2, 1:-2, 1:, vs.taup1] - vs.u[1:-2, 1:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskU[1:-2, 1:-2, 1:] * vs.maskU[1:-2, 1:-2, :-1])
        diss = update(diss, at[1:-2, 1:-2, :-1], (vs.u[1:-2, 1:-2, 1:, vs.tau] - vs.u[1:-2, 1:-2, :-1, vs.tau]) \
            * flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
        diss = update(diss, at[:, :, -1], 0.0)
        diss = numerics.ugrid_to_tgrid(state, diss)
        vs.K_diss_gm = diss

    if settings.enable_implicit_vert_friction:
        aloc = vs.v[:, :, :, vs.taup1]
    else:
        aloc = vs.v[:, :, :, vs.tau]

    # implicit vertical friction of zonal momentum by GM
    ks = np.maximum(vs.kbot[1:-2, 1:-2], vs.kbot[1:-2, 2:-1])
    _, water_mask, edge_mask = utilities.create_water_masks(ks, settings.nz)

    fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :] + vs.kappa_gm[1:-2, 2:-1, :])
    delta, a_tri, b_tri, c_tri = (np.zeros_like(vs.maskV[1:-2, 1:-2, :]) for _ in range(4))
    delta = update(delta, at[:, :, :-1], settings.dt_mom / vs.dzw[np.newaxis, np.newaxis, :-1] * \
        fxa[:, :, :-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
    delta = update(delta, at[-1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + delta[:, :, 1:-1] / vs.dzt[np.newaxis, np.newaxis, 1:-1] + \
        delta[:, :, :-2] / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / vs.dzt[-1])
    c_tri = update(c_tri, at[...], - delta / vs.dzt[np.newaxis, np.newaxis, :])

    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, aloc[1:-2, 1:-2, :], water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    vs.v = update(vs.v, at[1:-2, 1:-2, :, vs.taup1], np.where(water_mask, sol, vs.v[1:-2, 1:-2, :, vs.taup1]))
    vs.dv_mix = update_add(vs.dv_mix, at[1:-2, 1:-2, :], (vs.v[1:-2, 1:-2, :, vs.taup1] - aloc[1:-2, 1:-2, :]) / settings.dt_mom * water_mask)

    if settings.enable_conserve_energy:
        # diagnose dissipation
        diss = np.zeros_like(vs.maskV)
        fxa = 0.5 * (vs.kappa_gm[1:-2, 1:-2, :-1] + vs.kappa_gm[1:-2, 2:-1, :-1])
        flux_top = update(flux_top, at[1:-2, 1:-2, :-1], fxa * (vs.v[1:-2, 1:-2, 1:, vs.taup1] - vs.v[1:-2, 1:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskV[1:-2, 1:-2, 1:] * vs.maskV[1:-2, 1:-2, :-1])
        diss = update(diss, at[1:-2, 1:-2, :-1], (vs.v[1:-2, 1:-2, 1:, vs.tau] - vs.v[1:-2, 1:-2, :-1, vs.tau]) \
            * flux_top[1:-2, 1:-2, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
        diss = update(diss, at[:, :, -1], 0.0)
        diss = numerics.vgrid_to_tgrid(state, diss)
        vs.K_diss_gm = vs.K_diss_gm + diss

    return KernelOutput(u=vs.u, v=vs.v, du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_gm=vs.K_diss_gm)
