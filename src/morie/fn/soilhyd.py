# morie.fn -- function file (rootcoder007/morie)
"""Soil hydraulic pedotransfer functions, retention curves, infiltration, runoff, erosion, chemistry and texture."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "saxton_rawls",
    "van_genuchten",
    "infiltration",
    "scs_runoff",
    "rusle",
    "soil_chemistry",
    "soil_carbon",
    "usda_texture",
    "sobel_filter",
]


def saxton_rawls(sand: float, clay: float, om: float, *, density_factor: float = 1.0) -> RichResult:
    r"""Soil water characteristics from texture and organic matter (Saxton and Rawls 2006, Table 1).

    ``sand`` and ``clay`` are weight fractions, ``om`` percent organic
    matter.  Returns ``theta1500`` (wilting point), ``theta33`` (field
    capacity), ``theta_s`` (saturation), available water ``theta33 -
    theta1500``, air-entry tension ``psi_e`` (kPa), normal bulk density
    ``rho_n = 2.65 (1 - theta_s)``, ``B``, ``A``, ``lambda = 1/B`` and
    saturated conductivity ``ksat = 1930 (theta_s - theta33)^{3 - lambda}``
    (mm/h), with the density adjustments of eqs. 7-10 for ``density_factor``.

    References
    ----------
    Saxton, K. E. and Rawls, W. J. (2006). Soil water characteristic
    estimates by texture and organic matter for hydrologic solutions. *Soil
    Science Society of America Journal*, 70(5), 1569-1578.

    Examples
    --------
    >>> r = saxton_rawls(0.40, 0.20, 2.5)
    >>> round(r.theta1500, 6), round(r.theta33, 6)
    (0.137024, 0.27961)
    """
    S, C, OM = float(sand), float(clay), float(om)
    t15t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    t15 = t15t + (0.14 * t15t - 0.02)
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    t33 = t33t + (1.283 * t33t**2 - 0.374 * t33t - 0.015)
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    pet = -21.67 * S - 27.93 * C - 81.97 * ts33 + 71.12 * S * ts33 + 8.29 * C * ts33 + 14.05 * S * C + 27.16
    pe = pet + (0.02 * pet**2 - 0.113 * pet - 0.70)
    ts = t33 + ts33 - 0.097 * S + 0.043
    rho_n = (1.0 - ts) * 2.65
    rho_df = rho_n * density_factor
    ts_df = 1.0 - rho_df / 2.65
    t33_df = t33 - 0.2 * (ts - ts_df)
    B = (math.log(1500.0) - math.log(33.0)) / (math.log(t33) - math.log(t15))
    A = math.exp(math.log(33.0) + B * math.log(t33))
    lam = 1.0 / B
    ks = 1930.0 * (ts - t33) ** (3.0 - lam)
    return RichResult(
        payload={
            "theta1500": t15,
            "theta33": t33,
            "theta_s": ts,
            "available_water": t33 - t15,
            "psi_e": pe,
            "rho_n": rho_n,
            "B": B,
            "A": A,
            "lambda": lam,
            "ksat": ks,
            "theta_s_df": ts_df,
            "theta33_df": t33_df,
            "rho_df": rho_df,
        }
    )


def van_genuchten(
    h, theta_r: float, theta_s: float, alpha: float, n: float, *, ks: float | None = None, pore_l: float = 0.5
) -> RichResult:
    r"""Van Genuchten (1980) retention ``theta(h) = theta_r + (theta_s - theta_r)[1 + (alpha h)^n]^{-m}``, ``m = 1 - 1/n``.

    ``h`` is suction (positive); with ``ks`` the Mualem conductivity ``K =
    ks S_e^l [1 - (1 - S_e^{1/m})^m]^2`` is returned too (``S_e`` effective
    saturation).

    References
    ----------
    van Genuchten, M. Th. (1980). A closed-form equation for predicting the
    hydraulic conductivity of unsaturated soils. *Soil Science Society of
    America Journal*, 44(5), 892-898.

    Examples
    --------
    >>> r = van_genuchten([0.0, 100.0], 0.05, 0.45, 0.02, 1.5, ks=10.0)
    >>> [round(v, 6) for v in r.theta], round(r.K[0], 6)
    ([0.45, 0.305694], 10.0)
    """
    hs = [float(v) for v in (h if isinstance(h, (list, tuple)) else [h])]
    m = 1.0 - 1.0 / n
    se = [1.0 if v <= 0 else (1.0 + (alpha * v) ** n) ** (-m) for v in hs]
    out = {"theta": [theta_r + (theta_s - theta_r) * s for s in se], "Se": se}
    if ks is not None:
        out["K"] = [ks * s**pore_l * (1.0 - (1.0 - s ** (1.0 / m)) ** m) ** 2 for s in se]
    return RichResult(payload=out)


def infiltration(
    t,
    *,
    model: str = "horton",
    f0: float | None = None,
    fc: float | None = None,
    k: float | None = None,
    S: float | None = None,
    A: float | None = None,
    Ks: float | None = None,
    psi: float | None = None,
    dtheta: float | None = None,
) -> RichResult:
    r"""Infiltration rate and cumulative depth at times ``t``.

    ``horton`` (Horton 1940): ``f = fc + (f0 - fc) e^{-kt}``, ``F = fc t + (f0
    - fc)(1 - e^{-kt})/k``; ``philip`` (Philip 1957): ``F = S t^{1/2} + A t``,
    ``f = S t^{-1/2}/2 + A``; ``green_ampt`` (Green and Ampt 1911): ``F``
    solving ``Ks t = F - psi dtheta ln(1 + F/(psi dtheta))`` (Newton), ``f =
    Ks (1 + psi dtheta / F)``.

    References
    ----------
    Horton, R. E. (1940). An approach toward a physical interpretation of
    infiltration-capacity. *Soil Science Society of America Proceedings*, 5,
    399-417.
    Green, W. H. and Ampt, G. A. (1911). Studies on soil physics. *Journal of
    Agricultural Science*, 4(1), 1-24.

    Examples
    --------
    >>> round(infiltration([1.0], model="horton", f0=10.0, fc=2.0, k=1.0).rate[0], 6)
    4.943036
    """
    ts = [float(v) for v in (t if isinstance(t, (list, tuple)) else [t])]
    if model == "horton":
        rate = [fc + (f0 - fc) * math.exp(-k * v) for v in ts]
        cum = [fc * v + (f0 - fc) * (1.0 - math.exp(-k * v)) / k for v in ts]
    elif model == "philip":
        rate = [0.5 * S / math.sqrt(v) + A if v > 0 else float("inf") for v in ts]
        cum = [S * math.sqrt(v) + A * v for v in ts]
    elif model == "green_ampt":
        pd = psi * dtheta
        cum = []
        for v in ts:
            F = max(Ks * v, math.sqrt(2 * pd * Ks * v), 1e-12)
            for _ in range(200):
                g = F - pd * math.log(1.0 + F / pd) - Ks * v
                dF = g / (1.0 - pd / (pd + F))
                F -= dF
                if abs(dF) < 1e-14 * max(1.0, F):
                    break
            cum.append(F)
        rate = [Ks * (1.0 + pd / F) if F > 0 else float("inf") for F in cum]
    else:
        raise ValueError("model must be horton, philip or green_ampt")
    return RichResult(payload={"rate": rate, "cumulative": cum})


def scs_runoff(P, CN: float, *, ia_ratio: float = 0.2) -> RichResult:
    r"""SCS curve-number runoff (USDA-SCS 1972): ``Q = (P - Ia)^2 / (P - Ia + S)`` for ``P > Ia`` (mm).

    ``S = 25400 / CN - 254`` and ``Ia = ia_ratio S``; also the runoff
    coefficient ``Q / P``.

    References
    ----------
    USDA Soil Conservation Service (1972). *National Engineering Handbook,
    Section 4: Hydrology*. Washington, DC.

    Examples
    --------
    >>> round(scs_runoff([50.0], 80).runoff[0], 6)
    13.80248
    """
    Ps = [float(v) for v in (P if isinstance(P, (list, tuple)) else [P])]
    if not 0 < CN <= 100:
        raise ValueError("CN must be in (0, 100]")
    S = 25400.0 / CN - 254.0
    Ia = ia_ratio * S
    Q = [(p - Ia) ** 2 / (p - Ia + S) if p > Ia else 0.0 for p in Ps]
    return RichResult(
        payload={"runoff": Q, "coefficient": [q / p if p > 0 else 0.0 for q, p in zip(Q, Ps)], "S": S, "Ia": Ia}
    )


def rusle(R: float, K: float, C: float, P: float, *, slope_length: float, slope_pct: float) -> RichResult:
    r"""Soil loss ``A = R K LS C P`` with the LS factor of Wischmeier and Smith (1978).

    ``LS = (lambda / 22.13)^m (65.41 sin^2 theta + 4.56 sin theta + 0.065)``
    with slope length ``lambda`` (m), slope angle ``theta = atan(s/100)`` and
    ``m = 0.5`` (slope >= 5 %), 0.4 (3.5-5 %), 0.3 (1-3.5 %), 0.2 (< 1 %).

    References
    ----------
    Wischmeier, W. H. and Smith, D. D. (1978). *Predicting Rainfall Erosion
    Losses*. USDA Agriculture Handbook 537.
    Renard, K. G. et al. (1997). *Predicting Soil Erosion by Water: A Guide
    to Conservation Planning with RUSLE*. USDA Agriculture Handbook 703.

    Examples
    --------
    >>> round(rusle(100.0, 0.3, 0.2, 1.0, slope_length=22.13, slope_pct=9.0).LS, 6)
    0.999312
    """
    s = float(slope_pct)
    m = 0.5 if s >= 5 else 0.4 if s >= 3.5 else 0.3 if s >= 1 else 0.2
    th = math.atan(s / 100.0)
    LS = (slope_length / 22.13) ** m * (65.41 * math.sin(th) ** 2 + 4.56 * math.sin(th) + 0.065)
    return RichResult(payload={"A": R * K * LS * C * P, "LS": LS, "m": m})


def soil_chemistry(
    *, na: float, ca: float, mg: float, k: float = 0.0, h_al: float = 0.0, na_ex: float | None = None
) -> RichResult:
    r"""Sodicity and cation exchange measures.

    Sodium adsorption ratio ``SAR = Na / sqrt((Ca + Mg)/2)`` (solution
    concentrations in mmol_c/L; U.S. Salinity Laboratory Staff 1954); cation
    exchange capacity as the sum of exchangeable cations ``Ca + Mg + K + Na +
    H + Al`` (cmol_c/kg); exchangeable sodium percentage ``100 Na_ex / CEC``
    (``na_ex`` defaults to ``na``).

    References
    ----------
    U.S. Salinity Laboratory Staff (1954). *Diagnosis and Improvement of
    Saline and Alkali Soils*. USDA Agriculture Handbook 60.

    Examples
    --------
    >>> r = soil_chemistry(na=10.0, ca=4.0, mg=4.0, k=1.0, h_al=1.0)
    >>> round(r.sar, 6), r.cec, round(r.esp, 6)
    (5.0, 20.0, 50.0)
    """
    sar = na / math.sqrt((ca + mg) / 2.0)
    cec = ca + mg + k + na + h_al
    nx = na if na_ex is None else na_ex
    return RichResult(payload={"sar": sar, "cec": cec, "esp": 100.0 * nx / cec})


def soil_carbon(
    soc_pct: float,
    bulk_density: float,
    depth_cm: float,
    *,
    coarse_fraction: float = 0.0,
    clay_pct: float | None = None,
    silt_pct: float | None = None,
    aggregate_fractions=None,
    aggregate_diameters=None,
) -> RichResult:
    r"""Soil organic carbon stock, Pieri structure index and aggregate mean weight diameter.

    Stock (Mg C/ha) ``= SOC(%) BD(g/cm^3) depth(cm) (1 - coarse)``; with clay
    and silt percentages Pieri's (1992) structural stability index ``SI =
    1.724 SOC / (clay + silt) x 100``; with aggregate mass fractions and class
    mean diameters the mean weight diameter ``sum w_i x_i`` (van Bavel 1950).

    References
    ----------
    Pieri, C. (1992). *Fertility of Soils: A Future for Farming in the West
    African Savannah*. Springer, Berlin.
    van Bavel, C. H. M. (1950). Mean weight-diameter of soil aggregates as a
    statistical index of aggregation. *Soil Science Society of America
    Proceedings*, 14, 20-23.

    Examples
    --------
    >>> round(soil_carbon(1.0, 1.3, 30.0).stock, 6)
    39.0
    """
    out = {"stock": soc_pct * bulk_density * depth_cm * (1.0 - coarse_fraction)}
    if clay_pct is not None and silt_pct is not None:
        out["pieri_si"] = 1.724 * soc_pct / (clay_pct + silt_pct) * 100.0
    if aggregate_fractions is not None:
        w = [float(v) for v in aggregate_fractions]
        x = [float(v) for v in aggregate_diameters]
        out["mwd"] = ssum(a * b for a, b in zip(w, x)) / ssum(w)
    return RichResult(payload=out)


def usda_texture(sand: float, silt: float, clay: float) -> str:
    r"""USDA soil texture class from percentages (the NRCS texture triangle rules).

    Examples
    --------
    >>> usda_texture(40, 40, 20), usda_texture(10, 85, 5), usda_texture(20, 20, 60)
    ('loam', 'silt', 'clay')
    """
    s, si, c = float(sand), float(silt), float(clay)
    if abs(s + si + c - 100.0) > 1e-6:
        raise ValueError("sand + silt + clay must be 100")
    if si + 1.5 * c < 15:
        return "sand"
    if si + 1.5 * c >= 15 and si + 2 * c < 30:
        return "loamy sand"
    if (7 <= c < 20 and s > 52 and si + 2 * c >= 30) or (c < 7 and si < 50 and si + 2 * c >= 30):
        return "sandy loam"
    if 7 <= c < 27 and 28 <= si < 50 and s <= 52:
        return "loam"
    if (si >= 50 and 12 <= c < 27) or (50 <= si < 80 and c < 12):
        return "silt loam"
    if si >= 80 and c < 12:
        return "silt"
    if 20 <= c < 35 and si < 28 and s > 45:
        return "sandy clay loam"
    if 27 <= c < 40 and 20 < s <= 45:
        return "clay loam"
    if 27 <= c < 40 and s <= 20:
        return "silty clay loam"
    if c >= 35 and s > 45:
        return "sandy clay"
    if c >= 40 and si >= 40:
        return "silty clay"
    return "clay"


def sobel_filter(grid, *, res: float = 1.0) -> RichResult:
    r"""Sobel edge filter (Sobel and Feldman 1968): gradients ``Gx``, ``Gy`` and magnitude on interior cells.

    ``Gx = ((z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)) / (8 res)`` and ``Gy = ((z7 +
    2 z8 + z9) - (z1 + 2 z2 + z3)) / (8 res)`` on the 3 x 3 window (row-major
    from the north-west); edges ``nan``.

    Examples
    --------
    >>> round(sobel_filter([[0, 1, 2], [0, 1, 2], [0, 1, 2]]).magnitude[1][1], 6)
    1.0
    """
    G = [[float(v) for v in r] for r in grid]
    nr, nc = len(G), len(G[0])
    nan = float("nan")
    gx = [[nan] * nc for _ in range(nr)]
    gy = [[nan] * nc for _ in range(nr)]
    mag = [[nan] * nc for _ in range(nr)]
    for i in range(1, nr - 1):
        for j in range(1, nc - 1):
            z = [G[i + a][j + b] for a in (-1, 0, 1) for b in (-1, 0, 1)]
            x = ((z[2] + 2 * z[5] + z[8]) - (z[0] + 2 * z[3] + z[6])) / (8.0 * res)
            y = ((z[6] + 2 * z[7] + z[8]) - (z[0] + 2 * z[1] + z[2])) / (8.0 * res)
            gx[i][j], gy[i][j], mag[i][j] = x, y, math.hypot(x, y)
    return RichResult(payload={"gx": gx, "gy": gy, "magnitude": mag})


def cheatsheet() -> str:
    return (
        "saxton_rawls / van_genuchten / infiltration / scs_runoff / rusle / usda_texture -> soil hydrology and erosion."
    )
