from veros.core.operators import numpy as np

from veros import veros_routine, veros_kernel, KernelOutput
from veros.distributed import global_sum
from veros.variables import allocate
from veros.core import advection, diffusion, isoneutral, density, utilities
from veros.core.operators import update, update_add, at


@veros_kernel
def advect_tracer(state, tr):
    """
    calculate time tendency of a tracer due to advection
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee(state, tr)
    else:
        flux_east, flux_north, flux_top = advection.adv_flux_2nd(state, tr)

    dtr = allocate(state.dimensions, ("xt", "yt", "zt"))
    dtr = update(dtr, at[2:-2, 2:-2, :], vs.maskT[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                 / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                 - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                 / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))
    dtr = update_add(dtr, at[:, :, 0], -1 * vs.maskT[:, :, 0] * flux_top[:, :, 0] / vs.dzt[0])
    dtr = update_add(dtr, at[:, :, 1:], -1 * vs.maskT[:, :, 1:] * (flux_top[:, :, 1:] - flux_top[:, :, :-1]) / vs.dzt[1:])

    return dtr


@veros_kernel
def advect_temperature(state):
    """
    integrate temperature
    """
    vs = state.variables
    dtr = advect_tracer(state, vs.temp[..., vs.tau])
    vs.dtemp = update(vs.dtemp, at[..., vs.tau], dtr)
    return KernelOutput(dtemp=vs.dtemp)


@veros_kernel
def advect_salinity(state):
    """
    integrate salinity
    """
    vs = state.variables
    dtr = advect_tracer(state, vs.salt[..., vs.tau])
    vs.dsalt = update(vs.dsalt, at[..., vs.tau], dtr)
    return KernelOutput(dsalt=vs.dsalt)


@veros_kernel
def calc_eq_of_state(state, n):
    """
    calculate density, stability frequency, dynamic enthalpy and derivatives
    for time level n from temperature and salinity
    """
    vs = state.variables
    settings = state.settings

    salt = vs.salt[..., n]
    temp = vs.temp[..., n]
    press = np.abs(vs.zt)

    """
    calculate new density
    """
    vs.rho = update(vs.rho, at[..., n], density.get_rho(state, salt, temp, press) * vs.maskT)

    """
    calculate new potential density
    """
    vs.prho = update(vs.prho, at[...], density.get_potential_rho(state, salt, temp) * vs.maskT)

    """
    calculate new dynamic enthalpy and derivatives
    """
    if settings.enable_conserve_energy:
        vs.Hd = update(vs.Hd, at[..., n], density.get_dyn_enthalpy(state, salt, temp, press) * vs.maskT)
        vs.int_drhodT = update(vs.int_drhodT, at[..., n], density.get_int_drhodT(state, salt, temp, press))
        vs.int_drhodS = update(vs.int_drhodS, at[..., n], density.get_int_drhodS(state, salt, temp, press))

    """
    new stability frequency
    """
    fxa = -settings.grav / settings.rho_0 / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[:, :, :-1]
    vs.Nsqr = update(vs.Nsqr, at[:, :, :-1, n], fxa * (density.get_rho(
                                    state, salt[:, :, 1:], temp[:, :, 1:], press[:-1]
                                ) - vs.rho[:, :, :-1, n]))
    vs.Nsqr = update(vs.Nsqr, at[:, :, -1, n], vs.Nsqr[:, :, -2, n])

    return KernelOutput(rho=vs.rho, prho=vs.prho, Hd=vs.Hd, int_drhodT=vs.int_drhodT, int_drhodS=vs.int_drhodS, Nsqr=vs.Nsqr)


@veros_kernel
def advect_temp_salt_enthalpy(state):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    vs = state.variables
    settings = state.settings

    vs.dtemp = advect_temperature(state).dtemp
    vs.dsalt = advect_salinity(state).dsalt

    if settings.enable_conserve_energy:
        """
        advection of dynamic enthalpy
        """
        if settings.enable_superbee_advection:
            flux_east, flux_north, flux_top = advection.adv_flux_superbee(state, vs.Hd[:, :, :, vs.tau])
        else:
            flux_east, flux_north, flux_top = advection.adv_flux_2nd(state, vs.Hd[:, :, :, vs.tau])

        vs.dHd = update(vs.dHd, at[2:-2, 2:-2, :, vs.tau], vs.maskT[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                          / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dxt[2:-2, np.newaxis, np.newaxis])
                                                          - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                          / (vs.cost[np.newaxis, 2:-2, np.newaxis] * vs.dyt[np.newaxis, 2:-2, np.newaxis])))
        vs.dHd = update_add(vs.dHd, at[:, :, 0, vs.tau], -1 * vs.maskT[:, :, 0] \
            * flux_top[:, :, 0] / vs.dzt[0])
        vs.dHd = update_add(vs.dHd, at[:, :, 1:, vs.tau], -1 * vs.maskT[:, :, 1:] \
            * (flux_top[:, :, 1:] - flux_top[:, :, :-1]) \
            / vs.dzt[np.newaxis, np.newaxis, 1:])

        """
        changes in dyn. Enthalpy due to advection
        """
        diss = allocate(state.dimensions, ("xt", "yt", "zt"))
        diss = update(diss, at[2:-2, 2:-2, :], settings.grav / settings.rho_0 * (-vs.int_drhodT[2:-2, 2:-2, :, vs.tau] * vs.dtemp[2:-2, 2:-2, :, vs.tau]
                                              - vs.int_drhodS[2:-2, 2:-2, :, vs.tau] * vs.dsalt[2:-2, 2:-2, :, vs.tau]) \
                                           - vs.dHd[2:-2, 2:-2, :, vs.tau])

        """
        contribution by vertical advection is - g rho w / rho0, substract this also
        """
        diss = update_add(diss, at[:, :, :-1], -0.25 * settings.grav / settings.rho_0 * vs.w[:, :, :-1, vs.tau] \
            * (vs.rho[:, :, :-1, vs.tau] + vs.rho[:, :, 1:, vs.tau]) \
            * vs.dzw[np.newaxis, np.newaxis, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1])
        diss = update_add(diss, at[:, :, 1:], -0.25 * settings.grav / settings.rho_0 * vs.w[:, :, :-1, vs.tau] \
            * (vs.rho[:, :, 1:, vs.tau] + vs.rho[:, :, :-1, vs.tau]) \
            * vs.dzw[np.newaxis, np.newaxis, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])

    if settings.enable_conserve_energy and settings.enable_tke:
        """
        dissipation by advection interpolated on W-grid
        """
        P_diss_adv_w = diffusion.dissipation_on_wgrid(state, diss, vs.kbot)

        """
        distribute P_diss_adv_w over domain, prevent draining of TKE
        """
        fxa = np.sum(vs.area_t[2:-2, 2:-2, np.newaxis] * P_diss_adv_w[2:-2, 2:-2, :-1]
                     * vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1]) \
            + np.sum(0.5 * vs.area_t[2:-2, 2:-2] * vs.P_diss_adv[2:-2, 2:-2, -1]
                     * vs.dzw[-1] * vs.maskW[2:-2, 2:-2, -1])
        tke_mask = vs.tke[2:-2, 2:-2, :-1, vs.tau] > 0.
        fxb = np.sum(vs.area_t[2:-2, 2:-2, np.newaxis] * vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1] * tke_mask) \
            + np.sum(0.5 * vs.area_t[2:-2, 2:-2] * vs.dzw[-1] * vs.maskW[2:-2, 2:-2, -1])

        fxa = global_sum(fxa)
        fxb = global_sum(fxb)

        vs.P_diss_adv = update(vs.P_diss_adv, at[2:-2, 2:-2, :-1], fxa / fxb * tke_mask)
        vs.P_diss_adv = update(vs.P_diss_adv, at[2:-2, 2:-2, -1], fxa / fxb)

    """
    Adam Bashforth time stepping for advection
    """
    vs.temp = update(vs.temp, at[:, :, :, vs.taup1], vs.temp[:, :, :, vs.tau] + settings.dt_tracer \
        * ((1.5 + settings.AB_eps) * vs.dtemp[:, :, :, vs.tau]
           - (0.5 + settings.AB_eps) * vs.dtemp[:, :, :, vs.taum1]) * vs.maskT)
    vs.salt = update(vs.salt, at[:, :, :, vs.taup1], vs.salt[:, :, :, vs.tau] + settings.dt_tracer \
        * ((1.5 + settings.AB_eps) * vs.dsalt[:, :, :, vs.tau]
           - (0.5 + settings.AB_eps) * vs.dsalt[:, :, :, vs.taum1]) * vs.maskT)

    return KernelOutput(temp=vs.temp, salt=vs.salt, dtemp=vs.dtemp, dsalt=vs.dsalt, dHd=vs.dHd, P_diss_adv=vs.P_diss_adv)


@veros_kernel
def vertmix_tempsalt(state):
    """
    vertical mixing of temperature and salinity
    """
    vs = state.variables
    settings = state.settings

    vs.dtemp_vmix = update(vs.dtemp_vmix, at[...], vs.temp[:, :, :, vs.taup1])
    vs.dsalt_vmix = update(vs.dsalt_vmix, at[...], vs.salt[:, :, :, vs.taup1])

    a_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    b_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    c_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    d_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    delta = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]

    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    delta = update(delta, at[:, :, :-1], settings.dt_tracer / vs.dzw[np.newaxis, np.newaxis, :-1] \
        * vs.kappaH[2:-2, 2:-2, :-1])
    delta = update(delta, at[:, :, -1], 0.)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri = update(b_tri, at[:, :, 1:], 1 + (delta[:, :, 1:] + delta[:, :, :-1]) \
        / vs.dzt[np.newaxis, np.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[np.newaxis, np.newaxis, :]
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzt[np.newaxis, np.newaxis, :-1])
    d_tri = vs.temp[2:-2, 2:-2, :, vs.taup1]
    d_tri = update_add(d_tri, at[:, :, -1], settings.dt_tracer * vs.forc_temp_surface[2:-2, 2:-2] / vs.dzt[-1])

    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    vs.temp = update(vs.temp, at[2:-2, 2:-2, :, vs.taup1], np.where(water_mask, sol, vs.temp[2:-2, 2:-2, :, vs.taup1]))

    d_tri = vs.salt[2:-2, 2:-2, :, vs.taup1]
    d_tri = update_add(d_tri, at[:, :, -1], settings.dt_tracer * vs.forc_salt_surface[2:-2, 2:-2] / vs.dzt[-1])

    sol = utilities.solve_implicit(
        a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask
    )
    vs.salt = update(vs.salt, at[2:-2, 2:-2, :, vs.taup1], np.where(water_mask, sol, vs.salt[2:-2, 2:-2, :, vs.taup1]))

    vs.dtemp_vmix = (vs.temp[:, :, :, vs.taup1] - vs.dtemp_vmix) / settings.dt_tracer
    vs.dsalt_vmix = (vs.salt[:, :, :, vs.taup1] - vs.dsalt_vmix) / settings.dt_tracer

    """
    boundary exchange
    """
    vs.temp = update(vs.temp, at[..., vs.taup1], utilities.enforce_boundaries(vs.temp[..., vs.taup1], settings.enable_cyclic_x))
    vs.salt = update(vs.salt, at[..., vs.taup1], utilities.enforce_boundaries(vs.salt[..., vs.taup1], settings.enable_cyclic_x))

    return KernelOutput(dtemp_vmix=vs.dtemp_vmix, temp=vs.temp, dsalt_vmix=vs.dsalt_vmix, salt=vs.salt)


@veros_kernel
def surf_densityf(state):
    """
    surface density flux
    """
    vs = state.variables

    vs.forc_rho_surface = vs.maskT[:, :, -1] * (
        density.get_drhodT(state,
                           vs.salt[:, :, -1, vs.taup1],
                           vs.temp[:, :, -1, vs.taup1],
                           np.abs(vs.zt[-1])) * vs.forc_temp_surface
        + density.get_drhodS(state,
                             vs.salt[:, :, -1, vs.taup1],
                             vs.temp[:, :, -1, vs.taup1],
                             np.abs(vs.zt[-1])) * vs.forc_salt_surface
    )

    return KernelOutput(forc_rho_surface=vs.forc_rho_surface)


@veros_kernel
def diag_P_diss_v(state):
    vs = state.variables
    settings = state.settings

    vs.P_diss_v = update(vs.P_diss_v, at[...], 0.0)
    aloc = allocate(state.dimensions, ("xt", "yt", "zt"))

    if settings.enable_conserve_energy:
        """
        diagnose dissipation of dynamic enthalpy by vertical mixing
        """
        fxa = (-vs.int_drhodT[2:-2, 2:-2, 1:, vs.taup1] + vs.int_drhodT[2:-2, 2:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1]
        vs.P_diss_v = update_add(vs.P_diss_v, at[2:-2, 2:-2, :-1], -settings.grav / settings.rho_0 * fxa * vs.kappaH[2:-2, 2:-2, :-1] \
            * (vs.temp[2:-2, 2:-2, 1:, vs.taup1] - vs.temp[2:-2, 2:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1])
        fxa = (-vs.int_drhodS[2:-2, 2:-2, 1:, vs.taup1] + vs.int_drhodS[2:-2, 2:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1]
        vs.P_diss_v = update_add(vs.P_diss_v, at[2:-2, 2:-2, :-1], -settings.grav / settings.rho_0 * fxa * vs.kappaH[2:-2, 2:-2, :-1] \
            * (vs.salt[2:-2, 2:-2, 1:, vs.taup1] - vs.salt[2:-2, 2:-2, :-1, vs.taup1]) \
            / vs.dzw[np.newaxis, np.newaxis, :-1] * vs.maskW[2:-2, 2:-2, :-1])

        fxa = 2 * vs.int_drhodT[2:-2, 2:-2, -1, vs.taup1] / vs.dzw[-1]
        vs.P_diss_v = update_add(vs.P_diss_v, at[2:-2, 2:-2, -1], - settings.grav / settings.rho_0 * fxa * \
            vs.forc_temp_surface[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2, -1])
        fxa = 2 * vs.int_drhodS[2:-2, 2:-2, -1, vs.taup1] / vs.dzw[-1]
        vs.P_diss_v = update_add(vs.P_diss_v, at[2:-2, 2:-2, -1], - settings.grav / settings.rho_0 * fxa * \
            vs.forc_salt_surface[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2, -1])

    if settings.enable_conserve_energy:
        """
        determine effect due to nonlinear equation of state
        """
        aloc = update(aloc, at[:, :, :-1], vs.kappaH[:, :, :-1] * vs.Nsqr[:, :, :-1, vs.taup1])
        vs.P_diss_nonlin = update(vs.P_diss_nonlin, at[:, :, :-1], vs.P_diss_v[:, :, :-1] - aloc[:, :, :-1])
        vs.P_diss_v = update(vs.P_diss_v, at[:, :, :-1], aloc[:, :, :-1])
    else:
        """
        diagnose N^2 vs. kappaH, i.e. exchange of pot. energy with TKE
        """
        vs.P_diss_v = update(vs.P_diss_v, at[:, :, :-1], vs.kappaH[:, :, :-1] * vs.Nsqr[:, :, :-1, vs.taup1])
        vs.P_diss_v = update(vs.P_diss_v, at[:, :, -1], -vs.forc_rho_surface * vs.maskT[:, :, -1] * settings.grav / settings.rho_0)

    return KernelOutput(P_diss_v=vs.P_diss_v, P_diss_nonlin=vs.P_diss_nonlin)


@veros_routine
def thermodynamics(state):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    """
    Advection tendencies for temperature, salinity and dynamic enthalpy
    """
    vs = state.variables
    settings = state.settings

    vs.update(advect_temp_salt_enthalpy(state))

    """
    horizontal diffusion
    """
    with state.timers['isoneutral']:
        if settings.enable_hor_diffusion:
            vs.update(diffusion.tempsalt_diffusion(state))

        if settings.enable_biharmonic_mixing:
            vs.update(diffusion.tempsalt_biharmonic(state))

        """
        sources like restoring zones, etc
        """
        if settings.enable_tempsalt_sources:
            vs.update(diffusion.tempsalt_sources(state))

        """
        isopycnal diffusion
        """
        if settings.enable_neutral_diffusion:
            vs.P_diss_iso = update(vs.P_diss_iso, at[...], 0.0)
            vs.dtemp_iso = update(vs.dtemp_iso, at[...], 0.0)
            vs.dsalt_iso = update(vs.dsalt_iso, at[...], 0.0)

            vs.update(isoneutral.isoneutral_diffusion_pre(state))
            vs.update(isoneutral.isoneutral_diffusion(state, tr=vs.temp, istemp=True))
            vs.update(isoneutral.isoneutral_diffusion(state, tr=vs.salt, istemp=False))

            if settings.enable_skew_diffusion:
                vs.P_diss_skew = update(vs.P_diss_skew, at[...], 0.0)
                vs.update(isoneutral.isoneutral_skew_diffusion(state, tr=vs.temp, istemp=True))
                vs.update(isoneutral.isoneutral_skew_diffusion(state, tr=vs.salt, istemp=False))

    with state.timers['vmix']:
        vs.update(vertmix_tempsalt(state))

    with state.timers['eq_of_state']:
        vs.update(calc_eq_of_state(state, vs.taup1))

    """
    surface density flux
    """
    vs.update(surf_densityf(state))

    with state.timers['vmix']:
        vs.update(diag_P_diss_v(state))
