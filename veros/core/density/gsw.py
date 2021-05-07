from veros.core.operators import numpy as np

from veros import veros_kernel

"""
==========================================================================
  in-situ density, dynamic enthalpy and derivatives
  from Absolute Salinity and Conservative
  Temperature, using the computationally-efficient 48-term expression for
  density in terms of SA, CT and p (IOC et al., 2010).
==========================================================================
"""

v01 = 9.998420897506056e+2
v02 = 2.839940833161907e0
v03 = -3.147759265588511e-2
v04 = 1.181805545074306e-3
v05 = -6.698001071123802e0
v06 = -2.986498947203215e-2
v07 = 2.327859407479162e-4
v08 = -3.988822378968490e-2
v09 = 5.095422573880500e-4
v10 = -1.426984671633621e-5
v11 = 1.645039373682922e-7
v12 = -2.233269627352527e-2
v13 = -3.436090079851880e-4
v14 = 3.726050720345733e-6
v15 = -1.806789763745328e-4
v16 = 6.876837219536232e-7
v17 = -3.087032500374211e-7
v18 = -1.988366587925593e-8
v19 = -1.061519070296458e-11
v20 = 1.550932729220080e-10
v21 = 1.0e0
v22 = 2.775927747785646e-3
v23 = -2.349607444135925e-5
v24 = 1.119513357486743e-6
v25 = 6.743689325042773e-10
v26 = -7.521448093615448e-3
v27 = -2.764306979894411e-5
v28 = 1.262937315098546e-7
v29 = 9.527875081696435e-10
v30 = -1.811147201949891e-11
v31 = -3.303308871386421e-5
v32 = 3.801564588876298e-7
v33 = -7.672876869259043e-9
v34 = -4.634182341116144e-11
v35 = 2.681097235569143e-12
v36 = 5.419326551148740e-6
v37 = -2.742185394906099e-5
v38 = -3.212746477974189e-7
v39 = 3.191413910561627e-9
v40 = -1.931012931541776e-12
v41 = -1.105097577149576e-7
v42 = 6.211426728363857e-10
v43 = -1.119011592875110e-10
v44 = -1.941660213148725e-11
v45 = -1.864826425365600e-14
v46 = 1.119522344879478e-14
v47 = -1.200507748551599e-15
v48 = 6.057902487546866e-17
rho0 = 1024.0


@veros_kernel
def gsw_rho(sa, ct, p):
    """
     density as a function of T, S, and p
     sa     : Absolute Salinity                               [g/kg]
     ct     : Conservative Temperature                        [deg C]
     p      : sea pressure                                    [dbar]
    ==========================================================================
    """
    # convert scalar values if necessary
    sa, ct, p = np.asarray(sa), np.asarray(ct), np.asarray(p)
    sqrtsa = np.sqrt(sa)
    v_hat_denominator = v01 + ct * (v02 + ct * (v03 + v04 * ct)) \
        + sa * (v05 + ct * (v06 + v07 * ct)
                + sqrtsa * (v08 + ct * (v09 + ct * (v10 + v11 * ct)))) \
        + p * (v12 + ct * (v13 + v14 * ct) + sa * (v15 + v16 * ct)
               + p * (v17 + ct * (v18 + v19 * ct) + v20 * sa))
    v_hat_numerator = v21 + ct * (v22 + ct * (v23 + ct * (v24 + v25 * ct)))  \
        + sa * (v26 + ct * (v27 + ct * (v28 + ct * (v29 + v30 * ct))) + v36 * sa
                + sqrtsa * (v31 + ct * (v32 + ct * (v33 + ct * (v34 + v35 * ct))))) \
        + p * (v37 + ct * (v38 + ct * (v39 + v40 * ct))
               + sa * (v41 + v42 * ct) + p * (v43 + ct * (v44 + v45 * ct + v46 * sa)
                                              + p * (v47 + v48 * ct)))
    return v_hat_denominator / v_hat_numerator - rho0


@veros_kernel
def gsw_drhodT(sa, ct, p):
    """
    d/dT of density
    sa     : Absolute Salinity                               [g/kg]
    ct     : Conservative Temperature                        [deg C]
    p      : sea pressure                                    [dbar]
    ==========================================================================
    """
    p = np.asarray(p)  # convert scalar value if necessary
    a01 = 2.839940833161907e0
    a02 = -6.295518531177023e-2
    a03 = 3.545416635222918e-3
    a04 = -2.986498947203215e-2
    a05 = 4.655718814958324e-4
    a06 = 5.095422573880500e-4
    a07 = -2.853969343267241e-5
    a08 = 4.935118121048767e-7
    a09 = -3.436090079851880e-4
    a10 = 7.452101440691467e-6
    a11 = 6.876837219536232e-7
    a12 = -1.988366587925593e-8
    a13 = -2.123038140592916e-11
    a14 = 2.775927747785646e-3
    a15 = -4.699214888271850e-5
    a16 = 3.358540072460230e-6
    a17 = 2.697475730017109e-9
    a18 = -2.764306979894411e-5
    a19 = 2.525874630197091e-7
    a20 = 2.858362524508931e-9
    a21 = -7.244588807799565e-11
    a22 = 3.801564588876298e-7
    a23 = -1.534575373851809e-8
    a24 = -1.390254702334843e-10
    a25 = 1.072438894227657e-11
    a26 = -3.212746477974189e-7
    a27 = 6.382827821123254e-9
    a28 = -5.793038794625329e-12
    a29 = 6.211426728363857e-10
    a30 = -1.941660213148725e-11
    a31 = -3.729652850731201e-14
    a32 = 1.119522344879478e-14
    a33 = 6.057902487546866e-17

    sqrtsa = np.sqrt(sa)
    v_hat_denominator = v01 + ct * (v02 + ct * (v03 + v04 * ct)) + sa * (v05 + ct * (v06 + v07 * ct)
                                                                         + sqrtsa * (v08 + ct * (v09 + ct * (v10 + v11 * ct)))) + p * (v12 + ct * (v13 + v14 * ct) + sa * (v15 + v16 * ct)
                                                                                                                                       + p * (v17 + ct * (v18 + v19 * ct) + v20 * sa))

    v_hat_numerator = v21 + ct * (v22 + ct * (v23 + ct * (v24 + v25 * ct))) \
        + sa * (v26 + ct * (v27 + ct * (v28 + ct * (v29 + v30 * ct))) + v36 * sa
                + sqrtsa * (v31 + ct * (v32 + ct * (v33 + ct * (v34 + v35 * ct))))) + p * (v37 + ct * (v38 + ct * (v39 + v40 * ct))
                                                                                           + sa * (v41 + v42 * ct) + p * (v43 + ct * (v44 + v45 * ct + v46 * sa) + p * (v47 + v48 * ct)))

    dvhatden_dct = a01 + ct * (a02 + a03 * ct) + sa * (a04 + a05 * ct + sqrtsa * (a06 + ct * (a07 + a08 * ct))) \
        + p * (a09 + a10 * ct + a11 * sa + p * (a12 + a13 * ct))

    dvhatnum_dct = a14 + ct * (a15 + ct * (a16 + a17 * ct)) + sa * (a18 + ct * (a19 + ct * (a20 + a21 * ct))
                                                                    + sqrtsa * (a22 + ct * (a23 + ct * (a24 + a25 * ct)))) \
        + p * (a26 + ct * (a27 + a28 * ct) + a29 * sa + p * (a30 + a31 * ct + a32 * sa + a33 * p))

    rec_num = 1.0 / v_hat_numerator
    rho = rec_num * v_hat_denominator
    return (dvhatden_dct - dvhatnum_dct * rho) * rec_num


@veros_kernel
def gsw_drhodS(sa, ct, p):
    """
     d/dS of density
     sa     : Absolute Salinity                               [g/kg]
     ct     : Conservative Temperature                        [deg C]
     p      : sea pressure                                    [dbar]
    ==========================================================================
    """
    p = np.asarray(p)  # convert scalar value if necessary
    b01 = -6.698001071123802e0
    b02 = -2.986498947203215e-2
    b03 = 2.327859407479162e-4
    b04 = -5.983233568452735e-2
    b05 = 7.643133860820750e-4
    b06 = -2.140477007450431e-5
    b07 = 2.467559060524383e-7
    b08 = -1.806789763745328e-4
    b09 = 6.876837219536232e-7
    b10 = 1.550932729220080e-10
    b11 = -7.521448093615448e-3
    b12 = -2.764306979894411e-5
    b13 = 1.262937315098546e-7
    b14 = 9.527875081696435e-10
    b15 = -1.811147201949891e-11
    b16 = -4.954963307079632e-5
    b17 = 5.702346883314446e-7
    b18 = -1.150931530388857e-8
    b19 = -6.951273511674217e-11
    b20 = 4.021645853353715e-12
    b21 = 1.083865310229748e-5
    b22 = -1.105097577149576e-7
    b23 = 6.211426728363857e-10
    b24 = 1.119522344879478e-14

    sqrtsa = np.sqrt(sa)
    v_hat_denominator = v01 + ct * (v02 + ct * (v03 + v04 * ct)) + sa * (v05 + ct * (v06 + v07 * ct)
                                                                         + sqrtsa * (v08 + ct * (v09 + ct * (v10 + v11 * ct)))) + p * (v12 + ct * (v13 + v14 * ct) + sa * (v15 + v16 * ct)
                                                                                                                                       + p * (v17 + ct * (v18 + v19 * ct) + v20 * sa))

    v_hat_numerator = v21 + ct * (v22 + ct * (v23 + ct * (v24 + v25 * ct))) \
        + sa * (v26 + ct * (v27 + ct * (v28 + ct * (v29 + v30 * ct))) + v36 * sa
                + sqrtsa * (v31 + ct * (v32 + ct * (v33 + ct * (v34 + v35 * ct))))) \
        + p * (v37 + ct * (v38 + ct * (v39 + v40 * ct)) + sa * (v41 + v42 * ct) + p * (v43 + ct * (v44 + v45 * ct + v46 * sa)
                                                                                       + p * (v47 + v48 * ct)))

    dvhatden_dsa = b01 + ct * (b02 + b03 * ct) + sqrtsa * (b04 + ct * (b05 + ct * (b06 + b07 * ct))) \
        + p * (b08 + b09 * ct + b10 * p)

    dvhatnum_dsa = b11 + ct * (b12 + ct * (b13 + ct * (b14 + b15 * ct))) \
        + sqrtsa * (b16 + ct * (b17 + ct * (b18 + ct * (b19 + b20 * ct)))) \
        + b21 * sa + p * (b22 + ct * (b23 + b24 * p))

    rec_num = 1.0 / v_hat_numerator
    rho = rec_num * v_hat_denominator
    return (dvhatden_dsa - dvhatnum_dsa * rho) * rec_num


@veros_kernel
def gsw_drhodP(sa, ct, p):
    """
     d/dp of density
     sa     : Absolute Salinity                               [g/kg]
     ct     : Conservative Temperature                        [deg C]
     p      : sea pressure                                    [dbar]
    ==========================================================================
    """
    p = np.asarray(p)  # convert scalar value if necessary
    c01 = -2.233269627352527e-2
    c02 = -3.436090079851880e-4
    c03 = 3.726050720345733e-6
    c04 = -1.806789763745328e-4
    c05 = 6.876837219536232e-7
    c06 = -6.174065000748422e-7
    c07 = -3.976733175851186e-8
    c08 = -2.123038140592916e-11
    c09 = 3.101865458440160e-10
    c10 = -2.742185394906099e-5
    c11 = -3.212746477974189e-7
    c12 = 3.191413910561627e-9
    c13 = -1.931012931541776e-12
    c14 = -1.105097577149576e-7
    c15 = 6.211426728363857e-10
    c16 = -2.238023185750219e-10
    c17 = -3.883320426297450e-11
    c18 = -3.729652850731201e-14
    c19 = 2.239044689758956e-14
    c20 = -3.601523245654798e-15
    c21 = 1.817370746264060e-16
    pa2db = 1e-4

    sqrtsa = np.sqrt(sa)
    v_hat_denominator = v01 + ct * (v02 + ct * (v03 + v04 * ct)) + sa * (v05 + ct * (v06 + v07 * ct)
                                                                         + sqrtsa * (v08 + ct * (v09 + ct * (v10 + v11 * ct)))) + p * (v12 + ct * (v13 + v14 * ct) + sa * (v15 + v16 * ct)
                                                                                                                                       + p * (v17 + ct * (v18 + v19 * ct) + v20 * sa))

    v_hat_numerator = v21 + ct * (v22 + ct * (v23 + ct * (v24 + v25 * ct))) \
        + sa * (v26 + ct * (v27 + ct * (v28 + ct * (v29 + v30 * ct))) + v36 * sa
                + sqrtsa * (v31 + ct * (v32 + ct * (v33 + ct * (v34 + v35 * ct)))))   \
        + p * (v37 + ct * (v38 + ct * (v39 + v40 * ct)) + sa * (v41 + v42 * ct) + p * (v43 + ct * (v44 + v45 * ct + v46 * sa)
                                                                                       + p * (v47 + v48 * ct)))

    dvhatden_dp = c01 + ct * (c02 + c03 * ct) + sa * (c04 + c05 * ct) + \
        p * (c06 + ct * (c07 + c08 * ct) + c09 * sa)

    dvhatnum_dp = c10 + ct * (c11 + ct * (c12 + c13 * ct)) \
        + sa * (c14 + c15 * ct) + p * (c16 + ct * (c17 + c18 * ct + c19 * sa) + p * (c20 + c21 * ct))

    rec_num = 1.0 / v_hat_numerator
    rho = rec_num * v_hat_denominator
    return pa2db * (dvhatden_dp - dvhatnum_dp * rho) * rec_num


@veros_kernel
def gsw_dyn_enthalpy(sa, ct, p):
    """
     Calculates dynamic enthalpy of seawater using the computationally
     efficient 48-term expression for density in terms of SA, CT and p
     (IOC et al., 2010)

     A component due to the constant reference density in Boussinesq
     approximation is removed

     sa     : Absolute Salinity                               [g/kg]
     ct     : Conservative Temperature                        [deg C]
     p      : sea pressure                                    [dbar]
    ==========================================================================
    """
    p = np.asarray(p)  # convert scalar value if necessary
    db2pa = 1e4                             # factor to convert from dbar to Pa
    sqrtsa = np.sqrt(sa)
    a0 = v21 + ct * (v22 + ct * (v23 + ct * (v24 + v25 * ct))) \
        + sa * (v26 + ct * (v27 + ct * (v28 + ct * (v29 + v30 * ct))) + v36 * sa
                + sqrtsa * (v31 + ct * (v32 + ct * (v33 + ct * (v34 + v35 * ct)))))
    a1 = v37 + ct * (v38 + ct * (v39 + v40 * ct)) + sa * (v41 + v42 * ct)
    a2 = v43 + ct * (v44 + v45 * ct + v46 * sa)
    a3 = v47 + v48 * ct
    b0 = v01 + ct * (v02 + ct * (v03 + v04 * ct)) + sa * (v05 + ct * (v06 + v07 * ct)
                                                          + sqrtsa * (v08 + ct * (v09 + ct * (v10 + v11 * ct))))
    b1 = 0.5 * (v12 + ct * (v13 + v14 * ct) + sa * (v15 + v16 * ct))
    b2 = v17 + ct * (v18 + v19 * ct) + v20 * sa
    b1sq = b1 * b1
    sqrt_disc = np.sqrt(b1sq - b0 * b2)
    cn = a0 + (2 * a3 * b0 * b1 / b2 - a2 * b0) / b2
    cm = a1 + (4 * a3 * b1sq / b2 - a3 * b0 - 2 * a2 * b1) / b2
    ca = b1 - sqrt_disc
    cb = b1 + sqrt_disc
    part = (cn * b2 - cm * b1) / (b2 * (cb - ca))
    Hd = db2pa * (p * (a2 - 2.0 * a3 * b1 / b2 + 0.5 * a3 * p) / b2 + (cm / (2.0 * b2)) * np.log(1.0 + p * (2.0 * b1 + b2 * p) / b0)
                  + part * np.log(1.0 + (b2 * p * (cb - ca)) / (ca * (cb + b2 * p))))
    return Hd - p * db2pa / rho0


@veros_kernel
def gsw_dHdT(sa_in, ct_in, p):
    """
    d/dT of dynamic enthalpy, analytical derivative

    sa     : Absolute Salinity                               [g/kg]
    ct     : Conservative Temperature                        [deg C]
    p      : sea pressure                                    [dbar]
    """
    p = np.asarray(p)  # convert scalar value if necessary
    sa = np.maximum(1e-1, sa_in)  # prevent division by zero
    ct = np.maximum(-12, ct_in)  # prevent blowing up for values smaller than -15 degC
    t1 = v45 * ct
    t2 = 0.2e1 * t1
    t3 = v46 * sa
    t4 = 0.5 * v12
    t5 = v14 * ct
    t7 = ct * (v13 + t5)
    t8 = 0.5 * t7
    t11 = sa * (v15 + v16 * ct)
    t12 = 0.5 * t11
    t13 = t4 + t8 + t12
    t15 = v19 * ct
    t19 = v17 + ct * (v18 + t15) + v20 * sa
    t20 = 1.0 / t19
    t24 = v47 + v48 * ct
    t25 = 0.5 * v13
    t26 = 1.0 * t5
    t27 = sa * v16
    t28 = 0.5 * t27
    t29 = t25 + t26 + t28
    t33 = t24 * t13
    t34 = t19 ** 2
    t35 = 1.0 / t34
    t37 = v18 + 2.0 * t15
    t38 = t35 * t37
    t48 = ct * (v44 + t1 + t3)
    t57 = v40 * ct
    t59 = ct * (v39 + t57)
    t64 = t13 ** 2
    t68 = t20 * t29
    t71 = t24 * t64
    t74 = v04 * ct
    t76 = ct * (v03 + t74)
    t79 = v07 * ct
    t82 = np.sqrt(sa)
    t83 = v11 * ct
    t85 = ct * (v10 + t83)
    t92 = v01 + ct * (v02 + t76) + sa * (v05 + ct * (v06 + t79) + t82 * (v08 + ct * (v09 + t85)))
    t93 = v48 * t92
    t105 = v02 + t76 + ct * (v03 + 2.0 * t74) + sa * (v06 + 2.0 * t79 +
                                                      t82 * (v09 + t85 + ct * (v10 + 2.0 * t83)))
    t106 = t24 * t105
    t107 = v44 + t2 + t3
    t110 = v43 + t48
    t117 = t24 * t92
    t120 = 4.0 * t71 * t20 - t117 - 2.0 * t110 * t13
    t123 = v38 + t59 + ct * (v39 + 2.0 * t57) + sa * v42 + (4.0 *
                                                            v48 * t64 * t20 + 8.0 * t33 * t68 - 4.0 * t71 * t38 - t93 - t106
                                                            - 2.0 * t107 * t13 - 2.0 * t110 * t29) * t20 - t120 * t35 * t37
    t128 = t19 * p
    t130 = p * (1.0 * v12 + 1.0 * t7 + 1.0 * t11 + t128)
    t131 = 1.0 / t92
    t133 = 1.0 + t130 * t131
    t134 = np.log(t133)
    t143 = v37 + ct * (v38 + t59) + sa * (v41 + v42 * ct) + t120 * t20
    t152 = t37 * p
    t156 = t92 ** 2
    t165 = v25 * ct
    t167 = ct * (v24 + t165)
    t169 = ct * (v23 + t167)
    t175 = v30 * ct
    t177 = ct * (v29 + t175)
    t179 = ct * (v28 + t177)
    t185 = v35 * ct
    t187 = ct * (v34 + t185)
    t189 = ct * (v33 + t187)
    t199 = t13 * t20
    t217 = 2.0 * t117 * t199 - t110 * t92
    t234 = v21 + ct * (v22 + t169) + sa * (v26 + ct * (v27 + t179) + v36 *
                                           sa + t82 * (v31 + ct * (v32 + t189))) + t217 * t20
    t241 = t64 - t92 * t19
    t242 = np.sqrt(t241)
    t243 = 1.0 / t242
    t244 = t4 + t8 + t12 - t242
    t245 = 1.0 / t244
    t247 = t4 + t8 + t12 + t242 + t128
    t248 = 1.0 / t247
    t249 = t242 * t245 * t248
    t252 = 1.0 + 2.0 * t128 * t249
    t253 = np.log(t252)
    t254 = t243 * t253
    t259 = t234 * t19 - t143 * t13
    t264 = t259 * t20
    t272 = 2.0 * t13 * t29 - t105 * t19 - t92 * t37
    t282 = t128 * t242
    t283 = t244 ** 2
    t287 = t243 * t272 / 2.0
    t292 = t247 ** 2
    t305 = 0.1e5 * p * (v44 + t2 + t3 - 2.0 * v48 * t13 * t20
                        - 2.0 * t24 * t29 * t20 + 2.0 * t33 * t38 + 0.5 * v48 * p) * t20  \
        - 0.1e5 * p * (v43 + t48 - 2.0 * t33 * t20 + 0.5 * t24 * p) * t38 \
        + 0.5e4 * t123 * t20 * t134 - 0.5e4 * t143 * t35 * t134 * t37 \
        + 0.5e4 * t143 * t20 * (p * (1.0 * v13 + 2.0 * t5 + 1.0 * t27 + t152) * t131
                                - t130 / t156 * t105) / t133 \
        + 0.5e4 * ((v22 + t169 + ct * (v23 + t167 + ct * (v24 + 2.0 * t165))
                    + sa * (v27 + t179 + ct * (v28 + t177 + ct * (v29 + 2.0 * t175)) + t82 * (v32 + t189
                                                                                            + ct * (v33 + t187 + ct * (v34 + 2.0 * t185)))) + (2.0 * t93 * t199 + 2.0 * t106 *
                                                                                                                                                t199 + 2.0 * t117 * t68 - 2.0 * t117 * t13 * t35 * t37 - t107
                                                                                                                                                * t92 - t110 * t105) * t20 - t217 * t35 * t37) * t19 + t234 * t37
                - t123 * t13 - t143 * t29) * t20 * t254 - 0.5e4 * t259 * \
        t35 * t254 * t37 - 0.25e4 * t264 / t242 / t241 * t253 * t272 \
        + 0.5e4 * t264 * t243 * (2.0 * t152 * t249 + t128 *
                                t243 * t245 * t248 * t272 - 2.0 * t282 /
                                t283 * t248 * (t25 + t26 + t28 - t287)
                                - 2.0 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)) / t252

    return t305


@veros_kernel
def gsw_dHdS(sa_in, ct_in, p):
    """
    d/dS of dynamic enthalpy, analytical derivative
    sa     : Absolute Salinity                               [g/kg]
    ct     : Conservative Temperature                        [deg C]
    p      : sea pressure                                    [dbar]
    """
    p = np.asarray(p)  # convert scalar value if necessary
    sa = np.maximum(1e-1, sa_in)  # prevent division by zero
    ct = np.maximum(-12.0, ct_in)  # prevent blowing up for values smaller than -15 degC
    t1 = ct * v46
    t3 = v47 + v48 * ct
    t4 = 0.5 * v15
    t5 = v16 * ct
    t6 = 0.5 * t5
    t7 = t4 + t6
    t13 = v17 + ct * (v18 + v19 * ct) + v20 * sa
    t14 = 1.0 / t13
    t17 = 0.5 * v12
    t20 = ct * (v13 + v14 * ct)
    t21 = 0.5 * t20
    t23 = sa * (v15 + t5)
    t24 = 0.5 * t23
    t25 = t17 + t21 + t24
    t26 = t3 * t25
    t27 = t13 ** 2
    t28 = 1.0 / t27
    t29 = t28 * v20
    t39 = ct * (v44 + v45 * ct + v46 * sa)
    t48 = v42 * ct
    t49 = t14 * t7
    t52 = t25 ** 2
    t53 = t3 * t52
    t58 = ct * (v06 + v07 * ct)
    t59 = np.sqrt(sa)
    t66 = t59 * (v08 + ct * (v09 + ct * (v10 + v11 * ct)))
    t68 = v05 + t58 + 3.0 / 2.0 * t66
    t69 = t3 * t68
    t72 = v43 + t39
    t86 = v01 + ct * (v02 + ct * (v03 + v04 * ct)) + sa * (v05 + t58 + t66)
    t87 = t3 * t86
    t90 = 4.0 * t53 * t14 - t87 - 2.0 * t72 * t25
    t93 = v41 + t48 + (8.0 * t26 * t49 - 4.0 * t53 * t29 - t69 - 2.0 *
                       t1 * t25 - 2.0 * t72 * t7) * t14 - t90 * t28 * v20
    t98 = t13 * p
    t100 = p * (1.0 * v12 + 1.0 * t20 + 1.0 * t23 + t98)
    t101 = 1.0 / t86
    t103 = 1.0 + t100 * t101
    t104 = np.log(t103)
    t115 = v37 + ct * (v38 + ct * (v39 + v40 * ct)) + sa * (v41 + t48) + t90 * t14
    t123 = v20 * p
    t127 = t86 ** 2
    t142 = ct * (v27 + ct * (v28 + ct * (v29 + v30 * ct)))
    t143 = v36 * sa
    t151 = v31 + ct * (v32 + ct * (v33 + ct * (v34 + v35 * ct)))
    t152 = t59 * t151
    t158 = t25 * t14
    t174 = 2.0 * t87 * t158 - t72 * t86
    t189 = v21 + ct * (v22 + ct * (v23 + ct * (v24 + v25 * ct))) + \
        sa * (v26 + t142 + t143 + t152) + t174 * t14
    t196 = t52 - t86 * t13
    t197 = np.sqrt(t196)
    t198 = 1.0 / t197
    t199 = t17 + t21 + t24 - t197
    t200 = 1.0 / t199
    t202 = t17 + t21 + t24 + t197 + t98
    t203 = 1.0 / t202
    t204 = t197 * t200 * t203
    t207 = 1.0 + 2.0 * t98 * t204
    t208 = np.log(t207)
    t209 = t198 * t208
    t214 = t189 * t13 - t115 * t25
    t219 = t214 * t14
    t227 = 2.0 * t25 * t7 - t68 * t13 - t86 * v20
    t237 = t98 * t197
    t238 = t199 ** 2
    t242 = t198 * t227 / 2.0
    t247 = t202 ** 2
    t260 = 0.1e5 * p * (t1 - 2.0 * t3 * t7 * t14 + 2.0 * t26 * t29) * t14 \
        - 0.1e5 * p * (v43 + t39 - 2.0 * t26 * t14 + 0.5 * t3 * p) * t29 \
        + 0.5e4 * t93 * t14 * t104 - 0.5e4 * t115 * t28 * t104 * v20 \
        + 0.5e4 * t115 * t14 * (p * (1.0 * v15 + 1.0 * t5 + t123) * t101 - t100 / t127 * t68) / t103 \
        + 0.5e4 * ((v26 + t142 + t143 + t152 + sa * (v36 + 1.0 / t59 * t151 / 2.0)
                    + (2.0 * t69 * t158 + 2.0 * t87 * t49 - 2.0 * t87 *
                       t25 * t28 * v20 - t1 * t86 - t72 * t68) * t14
                    - t174 * t28 * v20) * t13 + t189 * v20 - t93 * t25 - t115 * t7) * t14 * t209 - 0.5e4 * t214 * t28 * t209 * v20 \
        - 0.25e4 * t219 / t197 / t196 * t208 * t227 + 0.5e4 * t219 * t198 * (2.0 * t123 * t204
                                                                             + t98 * t198 * t200 * t203 * t227 - 2.0 *
                                                                             t237 / t238 * t203 *
                                                                             (t4 + t6 - t242)
                                                                             - 2.0 * t237 * t200 / t247 * (t4 + t6 + t242 + t123)) / t207
    return t260
