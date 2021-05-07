from veros.core.operators import numpy as np

from veros import veros_kernel, veros_routine, KernelOutput, runtime_settings
from veros.variables import allocate
from veros.core import utilities, advection
from veros.core.operators import update, update_add, update_multiply, at


@veros_routine
def init_eke(state):
    """
    Initialize EKE
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_eke_leewave_dissipation:
        vs.hrms_k0 = np.maximum(settings.eke_hrms_k0_min, 2 / settings.pi * vs.eke_topo_hrms**2
                             / np.maximum(1e-12, vs.eke_topo_lam)**1.5)


@veros_routine
def set_eke_diffusivities(state):
    vs = state.variables
    settings = state.settings

    eke_diff_out = set_eke_diffusivities_kernel(state)
    vs.update(eke_diff_out)

    if settings.enable_TEM_friction:
        kappa_gm_out = update_kappa_gm(vs)
        vs.update(kappa_gm_out)


@veros_kernel
def update_kappa_gm(state):
    vs = state.variables

    kappa_gm = (
        vs.K_gm * np.minimum(0.01, vs.coriolis_t[..., np.newaxis]**2 / np.maximum(1e-9, vs.Nsqr[..., vs.tau])) * vs.maskW
    )
    return KernelOutput(kappa_gm=kappa_gm)


@veros_kernel
def set_eke_diffusivities_kernel(state):
    """
    set skew diffusivity K_gm and isopycnal diffusivity K_iso
    set also vertical viscosity if TEM formalism is chosen
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_eke:
        """
        calculate Rossby radius as minimum of mid-latitude and equatorial R. rad.
        """
        C_rossby = np.sum(np.sqrt(np.maximum(0., vs.Nsqr[:, :, :, vs.tau]))
                               * vs.dzw[np.newaxis, np.newaxis, :] * vs.maskW[:, :, :] / settings.pi, axis=2)
        vs.L_rossby = np.minimum(C_rossby / np.maximum(np.abs(vs.coriolis_t), 1e-16), np.sqrt(C_rossby / np.maximum(2 * vs.beta, 1e-16)))

        """
        calculate vertical viscosity and skew diffusivity
        """
        vs.sqrteke = np.sqrt(np.maximum(0., vs.eke[:, :, :, vs.tau]))
        vs.L_rhines = np.sqrt(vs.sqrteke / np.maximum(vs.beta[..., np.newaxis], 1e-16))
        eke_len = np.maximum(settings.eke_lmin, np.minimum(settings.eke_cross * vs.L_rossby[..., np.newaxis], settings.eke_crhin * vs.L_rhines))
        vs.K_gm = np.minimum(settings.eke_k_max, settings.eke_c_k * eke_len * vs.sqrteke)
    else:
        """
        use fixed GM diffusivity
        """
        vs.K_gm = update(vs.K_gm, at[...], settings.K_gm_0)

    if settings.enable_eke and settings.enable_eke_isopycnal_diffusion:
        vs.K_iso = update(vs.K_iso, at[...], vs.K_gm)
    else:
        vs.K_iso = update(vs.K_iso, at[...], settings.K_iso_0)  # always constant

    if not settings.enable_eke:
        return KernelOutput(K_gm=vs.K_gm, K_iso=vs.K_iso)

    return KernelOutput(L_rossby=vs.L_rossby, L_rhines=vs.L_rhines, eke_len=eke_len, sqrteke=vs.sqrteke, K_gm=vs.K_gm, K_iso=vs.K_iso)


@veros_routine
def integrate_eke(state):
    vs = state.variables
    vs.update(integrate_eke_kernel(state))


@veros_kernel
def integrate_eke_kernel(state):
    """
    integrate EKE equation on W grid
    """
    vs = state.variables
    settings = state.settings

    c_int = allocate(state.dimensions, ("xt", "yt", "zt"))

    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    """
    forcing by dissipation by lateral friction and GM using TRM formalism or skew diffusion
    """
    forc = vs.K_diss_h - vs.P_diss_skew

    if settings.enable_TEM_friction:
        forc = forc + vs.K_diss_gm

    """
    store transfer due to isopycnal and horizontal mixing from dyn. enthalpy
    by non-linear eq.of state either to EKE or to heat
    """
    if not settings.enable_store_cabbeling_heat:
        forc = forc - vs.P_diss_hmix - vs.P_diss_iso

    conditional_outputs = {}

    """
    coefficient for dissipation of EKE:
    by lee wave generation, Ri-dependent interior loss of balance and bottom friction
    """
    if settings.enable_eke_leewave_dissipation:
        """
        by lee wave generation
        """
        # TODO: replace these haxx with fancy indexing / take along axis
        vs.c_lee = np.zeros_like(vs.c_lee)
        ks = vs.kbot[2:-2, 2:-2] - 1
        ki = np.arange(settings.nz)[np.newaxis, np.newaxis, :]
        boundary_mask = (ks >= 0) & (ks < settings.nz - 1)
        full_mask = boundary_mask[:, :, np.newaxis] & (ki == ks[:, :, np.newaxis])
        fxa = np.maximum(0, vs.Nsqr[2:-2, 2:-2, :, vs.tau])**0.25
        fxa *= 1.5 * fxa / np.sqrt(np.maximum(1e-6, np.abs(vs.coriolis_t[2:-2, 2:-2, np.newaxis]))) - 2
        vs.c_lee = update(vs.c_lee, at[2:-2, 2:-2], boundary_mask * settings.c_lee0 * settings.hrms_k0[2:-2, 2:-2] \
                            * np.sum(np.sqrt(vs.sqrteke[2:-2, 2:-2, :]) * np.maximum(0, fxa)
                            / vs.dzw[np.newaxis, np.newaxis, :] * full_mask, axis=-1))

        """
        Ri-dependent dissipation by interior loss of balance
        """
        vs.c_Ri_diss = np.zeros_like(vs.c_Ri_diss)
        uz = (((vs.u[1:, 1:, 1:, vs.tau] - vs.u[1:, 1:, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskU[1:, 1:, :-1])**2
              + ((vs.u[:-1, 1:, 1:, vs.tau] - vs.u[:-1, 1:, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskU[:-1, 1:, :-1])**2) \
            / (vs.maskU[1:, 1:, :-1] + vs.maskU[:-1, 1:, :-1] + 1e-18)
        vz = (((vs.v[1:, 1:, 1:, vs.tau] - vs.v[1:, 1:, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskV[1:, 1:, :-1])**2
              + ((vs.v[1:, :-1, 1:, vs.tau] - vs.v[1:, :-1, :-1, vs.tau]) / vs.dzt[np.newaxis, np.newaxis, :-1] * vs.maskV[1:, :-1, :-1])**2) \
            / (vs.maskV[1:, 1:, :-1] + vs.maskV[1:, :-1, :-1] + 1e-18)
        Ri = np.maximum(1e-8, vs.Nsqr[1:, 1:, :-1, vs.tau]) / (uz + vz + 1e-18)
        fxa = 1 - 0.5 * (1. + np.tanh((Ri - settings.eke_Ri0) / settings.eke_Ri1))
        vs.c_Ri_diss = update(vs.c_Ri_diss, at[1:, 1:, :-1], vs.maskW[1:, 1:, :-1] * fxa * settings.eke_int_diss0)
        vs.c_Ri_diss = update(vs.c_Ri_diss, at[:, :, -1], vs.c_Ri_diss[:, :, -2] * vs.maskW[:, :, -1])

        """
        vertically integrate Ri-dependent dissipation and EKE
        """
        a_loc = np.sum(vs.c_Ri_diss[:, :, :-1] * vs.eke[:, :, :-1, vs.tau] * vs.maskW[:, :, :-1] * vs.dzw[:-1], axis=2)
        b_loc = np.sum(vs.eke[:, :, :-1, vs.tau] *
                       vs.maskW[:, :, :-1] * vs.dzw[:-1], axis=2)
        a_loc += vs.c_Ri_diss[:, :, -1] * vs.eke[:, :, -1, vs.tau] * vs.maskW[:, :, -1] * vs.dzw[-1] * 0.5
        b_loc += vs.eke[:, :, -1, vs.tau] * vs.maskW[:, :, -1] * vs.dzw[-1] * 0.5

        """
        add bottom fluxes by lee waves and bottom friction to a_loc
        """
        a_loc = update_add(a_loc, at[2:-2, 2:-2], np.sum((vs.c_lee[2:-2, 2:-2, np.newaxis] * vs.eke[2:-2, 2:-2, :, vs.tau]
                                     * vs.maskW[2:-2, 2:-2, :] * vs.dzw[np.newaxis, np.newaxis, :]
                                     + 2 * settings.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.tau]
                                     * np.sqrt(2) * vs.sqrteke[2:-2, 2:-2, :]
                                     * vs.maskW[2:-2, 2:-2, :]) * full_mask, axis=-1) * boundary_mask)

        """
        dissipation constant is vertically integrated forcing divided by
        vertically integrated EKE to account for vertical EKE radiation
        """
        mask = b_loc > 0
        a_loc = update(a_loc, at[...], np.where(mask, a_loc / (b_loc + 1e-20), 0.))
        c_int = update(c_int, at[...], a_loc[:, :, np.newaxis])

        conditional_outputs.update(c_lee=vs.c_lee, c_Ri_diss=vs.c_Ri_diss)
    else:
        """
        dissipation by local interior loss of balance with constant coefficient
        """
        c_int = settings.eke_c_eps * vs.sqrteke / vs.eke_len * vs.maskW

    """
    vertical diffusion of EKE,forcing and dissipation
    """
    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    delta, a_tri, b_tri, c_tri, d_tri = (np.zeros_like(vs.kappaM[2:-2, 2:-2, :]) for _ in range(5))
    delta = update(delta, at[:, :, :-1], settings.dt_tracer / vs.dzt[np.newaxis, np.newaxis, 1:] * 0.5 \
        * (vs.kappaM[2:-2, 2:-2, :-1] + vs.kappaM[2:-2, 2:-2, 1:]) * settings.alpha_eke)
    a_tri = update(a_tri, at[:, :, 1:-1], -delta[:, :, :-2] / vs.dzw[1:-1])
    a_tri = update(a_tri, at[:, :, -1], -delta[:, :, -2] / (0.5 * vs.dzw[-1]))
    b_tri = update(b_tri, at[:, :, 1:-1], 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / \
        vs.dzw[1:-1] + settings.dt_tracer * c_int[2:-2, 2:-2, 1:-1])
    b_tri = update(b_tri, at[:, :, -1], 1 + delta[:, :, -2] / \
        (0.5 * vs.dzw[-1]) + settings.dt_tracer * c_int[2:-2, 2:-2, -1])
    b_tri_edge = 1 + delta / vs.dzw[np.newaxis, np.newaxis, :] \
        + settings.dt_tracer * c_int[2:-2, 2:-2, :]
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzw[np.newaxis, np.newaxis, :-1])
    d_tri = update(d_tri, at[:, :, :], vs.eke[2:-2, 2:-2, :, vs.tau] + settings.dt_tracer * forc[2:-2, 2:-2, :])

    sol = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    vs.eke = update(vs.eke, at[2:-2, 2:-2, :, vs.taup1], np.where(water_mask, sol, vs.eke[2:-2, 2:-2, :, vs.taup1]))

    """
    store vs.eke dissipation
    """
    if settings.enable_eke_leewave_dissipation:
        vs.eke_diss_iw = np.zeros_like(vs.eke_diss_iw)
        vs.eke_diss_tke = vs.c_Ri_diss * vs.eke[:, :, :, vs.taup1]

        """
        flux by lee wave generation and bottom friction
        """
        vs.eke_diss_iw = update_add(vs.eke_diss_iw, at[2:-2, 2:-2, :], (vs.c_lee[2:-2, 2:-2, np.newaxis] * vs.eke[2:-2, 2:-2, :, vs.taup1]
                                       * vs.maskW[2:-2, 2:-2, :]) * full_mask)
        if runtime_settings.pyom_compatibility_mode:
            vs.eke_diss_tke = update_add(vs.eke_diss_tke, at[2:-2, 2:-2, :], (2 * settings.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.taup1] * np.sqrt(np.float32(2))
                                            * vs.sqrteke[2:-2, 2:-2, :] * vs.maskW[2:-2, 2:-2, :] / vs.dzw[np.newaxis, np.newaxis, :]) * full_mask)
        else:
            vs.eke_diss_tke = update_add(vs.eke_diss_tke, at[2:-2, 2:-2, :], (2 * settings.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.taup1] * np.sqrt(2)
                                            * vs.sqrteke[2:-2, 2:-2, :] * vs.maskW[2:-2, 2:-2, :] / vs.dzw[np.newaxis, np.newaxis, :]) * full_mask)
        """
        account for sligthly incorrect integral of dissipation due to time stepping
        """
        a_loc = np.sum((vs.eke_diss_iw[:, :, :-1] + vs.eke_diss_tke[:, :, :-1])
                       * vs.dzw[np.newaxis, np.newaxis, :-1], axis=2)
        b_loc = np.sum(c_int[:, :, :-1] * vs.eke[:, :, :-1, vs.taup1]
                       * vs.dzw[np.newaxis, np.newaxis, :-1], axis=2)
        a_loc += (vs.eke_diss_iw[:, :, -1] + vs.eke_diss_tke[:, :, -1]) * vs.dzw[-1] * 0.5
        b_loc += c_int[:, :, -1] * vs.eke[:, :, -1, vs.taup1] * vs.dzw[-1] * 0.5
        mask = a_loc != 0.
        b_loc = update(b_loc, at[...], np.where(mask, b_loc / (a_loc + 1e-20), 0.))
        vs.eke_diss_iw = update_multiply(vs.eke_diss_iw, at[...], b_loc[:, :, np.newaxis])
        vs.eke_diss_tke = update_multiply(vs.eke_diss_tke, at[...], b_loc[:, :, np.newaxis])

        """
        store diagnosed flux by lee waves and bottom friction
        """
        vs.eke_lee_flux = update(vs.eke_lee_flux, at[2:-2, 2:-2], np.where(boundary_mask, np.sum(vs.c_lee[2:-2, 2:-2, np.newaxis] * vs.eke[2:-2, 2:-2, :, vs.taup1]
                                                   * vs.dzw[np.newaxis, np.newaxis, :] * full_mask, axis=-1), vs.eke_lee_flux[2:-2, 2:-2]))
        vs.eke_bot_flux = update(vs.eke_bot_flux, at[2:-2, 2:-2], np.where(boundary_mask, np.sum(2 * settings.eke_r_bot * vs.eke[2:-2, 2:-2, :, vs.taup1]
                                                   * np.sqrt(2) * vs.sqrteke[2:-2, 2:-2, :] * full_mask, axis=-1), vs.eke_bot_flux[2:-2, 2:-2]))

        conditional_outputs.update(eke_bot_flux=vs.eke_bot_flux, eke_lee_flux=vs.eke_lee_flux)
    else:
        vs.eke_diss_iw = c_int * vs.eke[:, :, :, vs.taup1]
        vs.eke_diss_tke = np.zeros_like(vs.eke_diss_tke)

    """
    add tendency due to lateral diffusion
    """
    flux_east = update(flux_east, at[:-1, :, :], 0.5 * np.maximum(500., vs.K_gm[:-1, :, :] + vs.K_gm[1:, :, :]) \
        * (vs.eke[1:, :, :, vs.tau] - vs.eke[:-1, :, :, vs.tau]) \
        / (vs.cost[np.newaxis, :, np.newaxis] * vs.dxu[:-1, np.newaxis, np.newaxis]) * vs.maskU[:-1, :, :])
    flux_east = update(flux_east, at[-1, :, :], 0.)
    flux_north = update(flux_north, at[:, :-1, :], 0.5 * np.maximum(500., vs.K_gm[:, :-1, :] + vs.K_gm[:, 1:, :]) \
        * (vs.eke[:, 1:, :, vs.tau] - vs.eke[:, :-1, :, vs.tau]) \
        / vs.dyu[np.newaxis, :-1, np.newaxis] * vs.maskV[:, :-1, :] * vs.cosu[np.newaxis, :-1, np.newaxis])
    flux_north = update(flux_north, at[:, -1, :], 0.)
    vs.eke = update_add(vs.eke, at[2:-2, 2:-2, :, vs.taup1], settings.dt_tracer * vs.maskW[2:-2, 2:-2, :] \
        * ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
           + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))

    """
    add tendency due to advection
    """
    if settings.enable_eke_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(state, vs.eke[:, :, :, vs.tau])

    if settings.enable_eke_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(state, vs.eke[:, :, :, vs.tau])

    if settings.enable_eke_superbee_advection or settings.enable_eke_upwind_advection:
        vs.deke = update(vs.deke, at[2:-2, 2:-2, :, vs.tau], vs.maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                           - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                           / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))
        vs.deke = update_add(vs.deke, at[:, :, 0, vs.tau], -flux_top[:, :, 0] / vs.dzw[0])
        vs.deke = update_add(vs.deke, at[:, :, 1:-1, vs.tau], -(flux_top[:, :, 1:-1] -
                                   flux_top[:, :, :-2]) / vs.dzw[np.newaxis, np.newaxis, 1:-1])
        vs.deke = update_add(vs.deke, at[:, :, -1, vs.tau], -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * vs.dzw[-1]))
        """
        Adam Bashforth time stepping
        """
        vs.eke = update_add(vs.eke, at[:, :, :, vs.taup1], settings.dt_tracer * ((1.5 + settings.AB_eps) * vs.deke[:, :, :, vs.tau]
                                            - (0.5 + settings.AB_eps) * vs.deke[:, :, :, vs.taum1]))

        conditional_outputs.update(deke=vs.deke)

    return KernelOutput(eke=vs.eke, eke_diss_iw=vs.eke_diss_iw, eke_diss_tke=vs.eke_diss_tke, **conditional_outputs)
