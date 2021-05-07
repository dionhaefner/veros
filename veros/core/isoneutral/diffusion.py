from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine
from veros.variables import allocate
from veros.core import utilities, diffusion
from veros.core.operators import update, update_add, at


@veros_kernel
def _calc_tracer_fluxes(state, tr, K_iso, K_skew):
    vs = state.variables

    tr_pad = utilities.pad_z_edges(tr[..., vs.tau])

    K1 = K_iso - K_skew
    K2 = K_iso + K_skew

    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    """
    construct total isoneutral tracer flux at east face of 'T' cells
    """
    diffloc = np.zeros_like(K1)[1:-2, 2:-2]
    diffloc = update(diffloc, at[:, :, 1:], 0.25 * (K1[1:-2, 2:-2, 1:] + K1[1:-2, 2:-2, :-1] +
                                K1[2:-1, 2:-2, 1:] + K1[2:-1, 2:-2, :-1]))
    diffloc = update(diffloc, at[:, :, 0], 0.5 * (K1[1:-2, 2:-2, 0] + K1[2:-1, 2:-2, 0]))
    sumz = np.zeros_like(K1)[1:-2, 2:-2]
    for kr in range(2):
        for ip in range(2):
            sumz = sumz + diffloc * vs.Ai_ez[1:-2, 2:-2, :, ip, kr] * (
                tr_pad[1 + ip:-2 + ip, 2:-2, 1 + kr:-1 + kr or None] - tr_pad[1 + ip:-2 + ip, 2:-2, kr:-2 + kr])

    flux_east = update(flux_east, at[1:-2, 2:-2, :], sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :]) + (tr[2:-1, 2:-2, :, vs.tau] - tr[1:-2, 2:-2, :, vs.tau]) \
                             / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxu[1:-2, np.newaxis, np.newaxis]) * vs.K_11[1:-2, 2:-2, :])

    """
    construct total isoneutral tracer flux at north face of 'T' cells
    """
    diffloc = np.zeros_like(K1)[2:-2, 1:-2]
    diffloc = update(diffloc, at[:, :, 1:], 0.25 * (K1[2:-2, 1:-2, 1:] + K1[2:-2, 1:-2, :-1] +
                                K1[2:-2, 2:-1, 1:] + K1[2:-2, 2:-1, :-1]))
    diffloc = update(diffloc, at[:, :, 0], 0.5 * (K1[2:-2, 1:-2, 0] + K1[2:-2, 2:-1, 0]))
    sumz = np.zeros_like(K1)[2:-2, 1:-2]
    for kr in range(2):
        for jp in range(2):
            sumz = sumz + diffloc * vs.Ai_nz[2:-2, 1:-2, :, jp, kr] * (
                tr_pad[2:-2, 1 + jp:-2 + jp, 1 + kr:-1 + kr or None] - tr_pad[2:-2, 1 + jp:-2 + jp, kr:-2 + kr])

    flux_north = update(flux_north, at[2:-2, 1:-2, :], vs.cosu[np.newaxis, 1:-2, np.newaxis] * (sumz / (4. * vs.dzt[np.newaxis, np.newaxis, :])
                                + (tr[2:-2, 2:-1, :, vs.tau] - tr[2:-2, 1:-2, :, vs.tau]) / vs.dyu[np.newaxis, 1:-2, np.newaxis] * vs.K_22[2:-2, 1:-2, :]))

    """
    compute the vertical tracer flux 'flux_top' containing the K31
    and K32 components which are to be solved explicitly. The K33
    component will be treated implicitly. Note that there are some
    cancellations of dxu(i-1+ip) and dyu(jrow-1+jp)
    """
    diffloc = K2[2:-2, 2:-2, :-1]
    sumx = 0.
    for ip in range(2):
        for kr in range(2):
            sumx = sumx + diffloc * vs.Ai_bx[2:-2, 2:-2, :-1, ip, kr] / vs.cost[np.newaxis, 2:-2, np.newaxis] \
                * (tr[2 + ip:-2 + ip, 2:-2, kr:-1 + kr or None, vs.tau] - tr[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None, vs.tau])
    sumy = 0.
    for jp in range(2):
        for kr in range(2):
            sumy = sumy + diffloc * vs.Ai_by[2:-2, 2:-2, :-1, jp, kr] * vs.cosu[np.newaxis, 1 + jp:-3 + jp, np.newaxis] \
                * (tr[2:-2, 2 + jp:-2 + jp, kr:-1 + kr or None, vs.tau] - tr[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None, vs.tau])
    flux_top = update(flux_top, at[2:-2, 2:-2, :-1], sumx / (4 * vs.dxt[2:-2, np.newaxis, np.newaxis]) \
                              + sumy / (4 * vs.dyt[np.newaxis, 2:-2, np.newaxis] * vs.cost[np.newaxis, 2:-2, np.newaxis]))
    flux_top = update(flux_top, at[:, :, -1], 0.)

    return flux_east, flux_north, flux_top


@veros_kernel
def _calc_explicit_part(state, flux_east, flux_north, flux_top):
    vs = state.variables

    explicit_part = allocate(state.dimensions, ("xt", "yt", "zt"))
    explicit_part = update(explicit_part, at[2:-2, 2:-2, :], vs.maskT[2:-2, 2:-2, :] * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :]) / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                  + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :]) / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))
    explicit_part = update_add(explicit_part, at[:, :, 0], vs.maskT[:, :, 0] * flux_top[:, :, 0] / vs.dzt[0])
    explicit_part = update_add(explicit_part, at[:, :, 1:], vs.maskT[:, :, 1:] * \
        (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / \
        vs.dzt[np.newaxis, np.newaxis, 1:])

    return explicit_part


@veros_kernel
def _calc_implicit_part(state, tr):
    vs = state.variables
    settings = state.settings

    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    a_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    b_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    c_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    delta = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]

    delta = update(delta, at[:, :, :-1], settings.dt_tracer / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.K_33[2:-2, 2:-2, :-1])
    delta = update(delta, at[:, :, -1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) \
                        / vs.dzt[np.newaxis, np.newaxis, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / vs.dzt[np.newaxis, np.newaxis, -1])
    b_tri_edge = 1 + (delta[:, :, :] / vs.dzt[np.newaxis, np.newaxis, :])
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1])
    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, tr[2:-2, 2:-2, :, vs.taup1], water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    implicit_part = np.where(water_mask, sol, tr[2:-2, 2:-2, :, vs.taup1])
    return implicit_part


@veros_kernel(static_args=("iso", "skew"))
def isoneutral_diffusion_tracer(state, tr, dtracer_iso, iso=True, skew=False):
    """
    Isoneutral diffusion for general tracers
    """
    vs = state.variables
    settings = state.settings

    if iso:
        K_iso = vs.K_iso
    else:
        K_iso = np.zeros_like(vs.K_iso)
    if skew:
        K_skew = vs.K_gm
    else:
        K_skew = np.zeros_like(vs.K_gm)

    flux_east, flux_north, flux_top = _calc_tracer_fluxes(state, tr, K_iso, K_skew)

    """
    add explicit part
    """
    aloc = _calc_explicit_part(state, flux_east, flux_north, flux_top)
    dtracer_iso = dtracer_iso + aloc
    tr = update_add(tr, at[2:-2, 2:-2, :, vs.taup1], settings.dt_tracer * aloc[2:-2, 2:-2, :])

    """
    add implicit part
    """
    if iso:
        aloc = update(aloc, at[...], tr[:, :, :, vs.taup1])
        tr = update(tr, at[2:-2, 2:-2, :, vs.taup1], _calc_implicit_part(state, tr))
        dtracer_iso = dtracer_iso + (tr[:, :, :, vs.taup1] - aloc) / settings.dt_tracer

    return tr, dtracer_iso, flux_east, flux_north, flux_top


@veros_kernel(static_args=("istemp", "iso", "skew"))
def isoneutral_diffusion_kernel(state, tr, istemp, iso=True, skew=False):
    vs = state.variables
    settings = state.settings

    if istemp:
        dtracer_iso = vs.dtemp_iso
    else:
        dtracer_iso = vs.dsalt_iso

    tr, dtracer_iso, flux_east, flux_north, flux_top = isoneutral_diffusion_tracer(
        state, tr, dtracer_iso, iso=iso, skew=skew
    )

    """
    dissipation by isopycnal mixing
    """
    if settings.enable_conserve_energy:
        if istemp:
            int_drhodX = vs.int_drhodT[:, :, :, vs.tau]
        else:
            int_drhodX = vs.int_drhodS[:, :, :, vs.tau]

        """
        dissipation interpolated on W-grid
        """
        diss = diffusion.compute_dissipation(state, int_drhodX, flux_east, flux_north)
        diss_wgrid = diffusion.dissipation_on_wgrid(state, diss, vs.kbot)

        if not iso:
            P_diss_skew = vs.P_diss_skew + diss_wgrid
        else:
            P_diss_iso = vs.P_diss_iso + diss_wgrid

        """
        diagnose dissipation of dynamic enthalpy by explicit and implicit vertical mixing
        """
        fxa = (-int_drhodX[2:-2, 2:-2, 1:] + int_drhodX[2:-2, 2:-2, :-1]) / \
            vs.dzw[np.newaxis, np.newaxis, :-1]

        if istemp:
            tracer = vs.temp
        else:
            tracer = vs.salt

        if not iso:
            P_diss_skew = update_add(P_diss_skew, at[2:-2, 2:-2, :-1], - settings.grav / settings.rho_0 * \
                fxa * flux_top[2:-2, 2:-2, :-1] * vs.maskW[2:-2, 2:-2, :-1])

            return (tracer, dtracer_iso, P_diss_skew)

        else:
            P_diss_iso = update_add(P_diss_iso, at[2:-2, 2:-2, :-1], - settings.grav / settings.rho_0 * fxa * flux_top[2:-2, 2:-2, :-1] * vs.maskW[2:-2, 2:-2, :-1] \
                - settings.grav / settings.rho_0 * fxa * vs.K_33[2:-2, 2:-2, :-1] * (tracer[2:-2, 2:-2, 1:, vs.taup1]
                                                                - tracer[2:-2, 2:-2, :-1, vs.taup1]) \
                / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1])

            return (tracer, dtracer_iso, P_diss_iso)


@veros_routine
def isoneutral_diffusion(state, tr, istemp):
    """
    Isopycnal diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_iso
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    vs = state.variables
    tracer, dtracer_iso, P_diss_iso = isoneutral_diffusion_kernel(state, tr, istemp, iso=True, skew=False)

    if istemp:
        vs.update(temp=tracer, dtemp_iso=dtracer_iso)
    else:
        vs.update(salt=tracer, dsalt_iso=dtracer_iso)

    vs.update(P_diss_iso=P_diss_iso)


@veros_routine
def isoneutral_skew_diffusion(state, tr, istemp):
    """
    Isopycnal skew diffusion for tracer,
    following functional formulation by Griffies et al
    Dissipation is calculated and stored in P_diss_skew
    T/S changes are added to dtemp_iso/dsalt_iso
    """
    vs = state.variables
    tracer, dtracer_iso, P_diss_skew = isoneutral_diffusion_kernel(state, tr, istemp, iso=False, skew=True)

    if istemp:
        vs.update(temp=tracer, dtemp_iso=dtracer_iso)
    else:
        vs.update(salt=tracer, dsalt_iso=dtracer_iso)

    vs.update(P_diss_skew=P_diss_skew)
