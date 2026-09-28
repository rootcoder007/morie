# morie.fn -- function file (rootcoder007/morie)
"""Marine and coastal remote sensing and ocean physics: OCx band-ratio chlorophyll, the
vertically generalised production model, euphotic depth, air-sea CO2 flux, oxygen solubility,
practical salinity (PSS-78), split-window sea-surface temperature, sediment transport by the
Shields criterion and Meyer-Peter-Mueller, Lagrangian oil-spill drift, a continuous-source
coastal plume, Lyzenga's depth-invariant bottom index and the floating-algae, mangrove and
bare-soil spectral indices."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal

__all__ = [
    "ocean_chlorophyll",
    "vgpm_production",
    "euphotic_depth",
    "co2_flux",
    "oxygen_solubility",
    "practical_salinity",
    "split_window_sst",
    "sediment_transport",
    "oil_spill_drift",
    "plume_concentration",
    "depth_invariant_index",
    "floating_algae_index",
    "mangrove_vegetation_index",
    "bare_soil_index",
]

_OCX = {
    "OC4": [0.3272, -2.9940, 2.7218, -1.2259, -0.5683],
    "OC4v4": [0.366, -3.067, 1.930, 0.649, -1.532],
    "OC3M": [0.2424, -2.7423, 1.8017, 0.0015, -1.2280],
    "OC2": [0.2511, -2.0853, 1.5035, -3.1747, 0.3383],
}


_K0 = {
    "mol/L": [-58.0931, 90.5069, 22.2940, 0.027766, -0.025888, 0.0050578],
    "mol/kg": [-60.2409, 93.4517, 23.3585, 0.023517, -0.023656, 0.0047036],
}


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _one(x):
    return isinstance(x, (int, float))


def _map(f, *args):
    if all(_one(a) for a in args):
        return f(*[float(a) for a in args])
    vs = [_vec(a) if not _one(a) else None for a in args]
    n = max(len(v) for v in vs if v is not None)
    cols = [v if v is not None else [float(a)] * n for v, a in zip(vs, args)]
    return [f(*row) for row in zip(*cols)]


def ocean_chlorophyll(blue, green, algorithm="OC4", coefficients=None):
    r"""Band-ratio chlorophyll-a (mg/m^3) of the NASA OCx family.

    ``log10 chl = a0 + sum_{i=1}^4 a_i x^i`` with ``x = log10(max(Rrs_blue) /
    Rrs_green)``: ``blue`` holds one or more blue remote-sensing reflectances
    per pixel (rows = pixels) and the maximum is taken (maximum band ratio).
    Coefficients: ``"OC4"`` SeaWiFS OC4 (O'Reilly et al. 2019 reprocessing),
    ``"OC4v4"`` (O'Reilly et al. 2000), ``"OC3M"`` MODIS, ``"OC2"``; or give
    ``coefficients``.

    References
    ----------
    O'Reilly, J. E. et al. (1998). Ocean color chlorophyll algorithms for
    SeaWiFS. *Journal of Geophysical Research* 103, 24937-24953.

    Examples
    --------
    >>> round(ocean_chlorophyll([[0.004, 0.005, 0.0045]], [0.003], "OC4v4")[0], 10)
    0.6080701892
    """
    a = list(coefficients) if coefficients is not None else _OCX[algorithm]
    B = np.asarray(blue, dtype=float)
    rows = [[float(v)] for v in B.ravel().tolist()] if B.ndim == 1 else [[float(v) for v in r] for r in B.tolist()]
    g = _vec(green)
    out = []
    for r, gg in zip(rows, g):
        x = math.log10(max(r) / gg)
        out.append(10 ** ssum(a[i] * x**i for i in range(len(a))))
    return out


def euphotic_depth(kd=None, secchi=None, fraction=0.01):
    r"""Euphotic (photic) zone depth ``z_eu = -ln(fraction) / K_d`` (1 percent light level).

    ``K_d`` is the diffuse attenuation coefficient (1/m); with a Secchi depth
    ``z_SD`` it is estimated as ``1.7 / z_SD`` (Poole and Atkins 1929).

    References
    ----------
    Kirk, J. T. O. (2011). *Light and Photosynthesis in Aquatic Ecosystems*,
    3rd edn. Cambridge University Press.

    Examples
    --------
    >>> round(euphotic_depth(secchi=10.0), 10)
    27.0892363882
    """
    if kd is None:
        if secchi is None:
            raise ValueError("give kd or secchi")
        kd = _map(lambda s: 1.7 / s, secchi)
    return _map(lambda k: -math.log(fraction) / k, kd)


def _pbopt(t):
    if t < -10.0:
        return 0.0
    if t < -1.0:
        return 1.13
    if t > 28.5:
        return 4.0
    c = [1.2956, 2.749e-1, 6.17e-2, -2.05e-2, 2.462e-3, -1.348e-4, 3.4132e-6, -3.27e-8]
    return ssum(c[i] * t**i for i in range(8))


def vgpm_production(chl, par, sst, day_length):
    r"""Net primary production (mg C m^-2 d^-1) by the VGPM of Behrenfeld and Falkowski (1997).

    ``PP = P_opt^B(T) chl DL 0.66125 E0 / (E0 + 4.1) z_eu`` with the 7th-order
    temperature polynomial for ``P_opt^B`` (1.13 below -1 C, 4 above 28.5 C),
    ``E0`` the daily PAR (mol photons m^-2 d^-1), ``DL`` day length (h) and
    ``z_eu`` from total chlorophyll (Morel and Berthon 1989): ``chl_tot = 38
    chl^0.425`` (chl < 1) else ``40.2 chl^0.507``; ``z_eu = 200 chl_tot^-0.293``,
    or ``568.2 chl_tot^-0.746`` when that is at most 102 m.

    References
    ----------
    Behrenfeld, M. J. and Falkowski, P. G. (1997). Photosynthetic rates derived
    from satellite-based chlorophyll concentration. *Limnology and Oceanography*
    42, 1-20.

    Examples
    --------
    >>> round(vgpm_production(0.5, 40.0, 15.0, 12.0), 8)
    959.28482621
    """

    def one(c, e, t, d):
        tot = 38.0 * c**0.425 if c < 1.0 else 40.2 * c**0.507
        z = 200.0 * tot**-0.293
        if z <= 102.0:
            z = 568.2 * tot**-0.746
        return _pbopt(t) * c * d * 0.66125 * e / (e + 4.1) * z

    return _map(one, chl, par, sst, day_length)


def co2_flux(u10, sst, sss, pco2_water, pco2_air, k0_units="mol/L"):
    r"""Air-sea CO2 flux (mmol m^-2 d^-1, positive out of the ocean) by the bulk formula.

    ``F = 0.24 k K0 (pCO2_w - pCO2_a)`` with ``k = 0.251 U10^2 (Sc/660)^-0.5``
    cm/h (Wanninkhof 2014), the CO2 Schmidt number ``Sc = 2116.8 - 136.25 T +
    4.7353 T^2 - 0.092307 T^3 + 0.0007555 T^4`` and the solubility ``K0``
    (mol L^-1 atm^-1) of Weiss (1974); pCO2 in microatm. ``k0_units="mol/kg"``
    uses Weiss's gravimetric coefficients (``K0`` in mol kg^-1 atm^-1, then
    the flux is per kg of seawater volume-equivalent, as in ``seacarb::K0``).

    References
    ----------
    Wanninkhof, R. (2014). Relationship between wind speed and gas exchange
    over the ocean revisited. *Limnology and Oceanography: Methods* 12,
    351-362.

    Weiss, R. F. (1974). Carbon dioxide in water and seawater: the solubility of
    a non-ideal gas. *Marine Chemistry* 2, 203-215.

    Examples
    --------
    >>> r = co2_flux(7.0, 20.0, 35.0, 420.0, 400.0)
    >>> round(r.flux, 10), round(r.schmidt, 10)
    (1.9485890784, 668.344)
    """

    def one(u, t, s, pw, pa):
        sc = 2116.8 - 136.25 * t + 4.7353 * t**2 - 0.092307 * t**3 + 0.0007555 * t**4
        k = 0.251 * u * u * (sc / 660.0) ** -0.5
        tk = (t + 273.15) / 100.0
        c = _K0[k0_units]
        k0 = math.exp(c[0] + c[1] / tk + c[2] * math.log(tk) + s * (c[3] + c[4] * tk + c[5] * tk * tk))
        return 0.24 * k * k0 * (pw - pa), k, sc, k0

    res = _map(lambda *a: one(*a), u10, sst, sss, pco2_water, pco2_air)
    if isinstance(res, tuple):
        f, k, sc, k0 = res
    else:
        f, k, sc, k0 = (list(z) for z in zip(*res))
    return RichResult(payload={"flux": f, "transfer_velocity": k, "schmidt": sc, "k0": k0})


def oxygen_solubility(temperature, salinity, units="umol/kg"):
    r"""Oxygen solubility of seawater (Garcia and Gordon 1992, Benson-Krause fit).

    ``ln C = A0 + A1 Ts + ... + A5 Ts^5 + S (B0 + B1 Ts + B2 Ts^2 + B3 Ts^3) +
    C0 S^2`` with ``Ts = ln((298.15 - t) / (273.15 + t))``, ``t`` the potential
    temperature on the IPTS-68 scale of the fit (``1.00024`` times the ITS-90
    input, as in ``gsw_O2sol_SP_pt``); ``units`` ``"umol/kg"`` or ``"ml/l"``. Percent saturation
    of an observed concentration is ``100 O2 / C``.

    References
    ----------
    Garcia, H. E. and Gordon, L. I. (1992). Oxygen solubility in seawater:
    better fitting equations. *Limnology and Oceanography* 37, 1307-1312.

    Examples
    --------
    >>> round(oxygen_solubility(10.0, 35.0), 10)
    274.5956644859
    """
    if units == "umol/kg":
        A = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
        B = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
        C0 = -2.75915e-7
    elif units == "ml/l":
        A = [2.00907, 3.22014, 4.05010, 4.94457, -2.56847e-1, 3.88767]
        B = [-6.24523e-3, -7.37614e-3, -1.03410e-2, -8.17083e-3]
        C0 = -4.88682e-7
    else:
        raise ValueError("units must be 'umol/kg' or 'ml/l'")

    def one(t90, s):
        t = 1.00024 * t90
        ts = math.log((298.15 - t) / (273.15 + t))
        return math.exp(ssum(A[i] * ts**i for i in range(6)) + s * ssum(B[i] * ts**i for i in range(4)) + C0 * s * s)

    return _map(one, temperature, salinity)


def practical_salinity(conductivity, temperature, pressure=0.0):
    r"""Practical salinity (PSS-78) from conductivity (mS/cm), ITS-90 temperature (C) and pressure (dbar).

    ``R = C / 42.914``; ``R_t = R / (R_p r_t)`` with the UNESCO (1983)
    polynomials ``r_t(t_68)`` and ``R_p``; ``S = sum a_i R_t^{i/2} + (t - 15) /
    (1 + 0.0162 (t - 15)) sum b_i R_t^{i/2}`` (``t_68 = 1.00024 t_90``). Valid
    for ``2 <= S <= 42``.

    References
    ----------
    UNESCO (1983). Algorithms for computation of fundamental properties of
    seawater. *UNESCO Technical Papers in Marine Science* 44.

    Examples
    --------
    >>> round(practical_salinity(42.914, 15.0 / 1.00024, 0.0), 6)
    35.0
    """
    a = [0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081]
    b = [0.0005, -0.0056, -0.0066, -0.0375, 0.0636, -0.0144]
    c = [0.6766097, 2.00564e-2, 1.104259e-4, -6.9698e-7, 1.0031e-9]

    def one(cond, t90, p):
        t = 1.00024 * t90
        R = cond / 42.914
        rt = ssum(c[i] * t**i for i in range(5))
        Rp = 1 + p * (2.070e-5 + p * (-6.370e-10 + p * 3.989e-15)) / (
            1 + 3.426e-2 * t + 4.464e-4 * t * t + (4.215e-1 - 3.107e-3 * t) * R
        )
        Rt = R / (Rp * rt)
        r = math.sqrt(Rt)
        ds = (t - 15) / (1 + 0.0162 * (t - 15)) * ssum(b[i] * r**i for i in range(6))
        return ssum(a[i] * r**i for i in range(6)) + ds

    return _map(one, conductivity, temperature, pressure)


def split_window_sst(t11, t12, coefficients, sst_first_guess=None, zenith=0.0):
    r"""Split-window sea-surface temperature (NLSST form).

    ``SST = a0 + a1 T11 + a2 (T11 - T12) T_sfc + a3 (T11 - T12) (sec theta - 1)``
    with brightness temperatures ``T11``, ``T12`` of the 11 and 12 micron
    channels, a first-guess ``T_sfc`` (default ``T11``; the MCSST form with
    ``T_sfc = 1``) and satellite zenith ``theta`` (degrees). The coefficients
    are sensor specific and supplied by the user.

    References
    ----------
    Walton, C. C., Pichel, W. G., Sapper, J. F. and May, D. A. (1998). The
    development and operational application of nonlinear algorithms for the
    measurement of sea surface temperatures with the NOAA polar-orbiting
    environmental satellites. *Journal of Geophysical Research* 103,
    27999-28012.

    Examples
    --------
    >>> round(split_window_sst(290.0, 289.2, [-260.0, 0.9, 0.08, 0.7], 18.0, 30.0), 10)
    2.2386323015
    """
    a0, a1, a2, a3 = [float(v) for v in coefficients]

    def one(x, y, g, z):
        g = x if math.isnan(g) else g
        return a0 + a1 * x + a2 * (x - y) * g + a3 * (x - y) * (1 / math.cos(math.radians(z)) - 1)

    fg = math.nan if sst_first_guess is None else sst_first_guess
    return _map(one, t11, t12, fg, zenith)


def sediment_transport(tau, d, rho_s=2650.0, rho=1025.0, nu=1.36e-6, g=9.81):
    r"""Shields mobility, Soulsby-Whitehouse critical Shields number and Meyer-Peter-Mueller bedload.

    ``theta = tau / ((rho_s - rho) g d)``, ``D* = d ((s - 1) g / nu^2)^(1/3)``,
    ``theta_cr = 0.30 / (1 + 1.2 D*) + 0.055 (1 - exp(-0.020 D*))`` and the
    volumetric bedload per unit width ``q = 8 (theta - theta_cr)^1.5 sqrt((s -
    1) g d^3)`` (0 below the threshold), ``s = rho_s / rho``; SI units.

    References
    ----------
    Soulsby, R. L. and Whitehouse, R. J. S. (1997). Threshold of sediment motion
    in coastal environments. *Pacific Coasts and Ports '97*, 149-154.

    Meyer-Peter, E. and Mueller, R. (1948). Formulas for bed-load transport.
    *Proceedings of the 2nd IAHR Congress*, Stockholm, 39-64.

    Examples
    --------
    >>> r = sediment_transport(1.2, 0.0005)
    >>> round(r.theta, 10), round(r.theta_cr, 10)
    (0.1505528111, 0.0328460664)
    """
    s = rho_s / rho

    def one(t, dd):
        th = t / ((rho_s - rho) * g * dd)
        ds = dd * ((s - 1) * g / (nu * nu)) ** (1.0 / 3.0)
        tc = 0.30 / (1 + 1.2 * ds) + 0.055 * (1 - math.exp(-0.020 * ds))
        q = 8.0 * (th - tc) ** 1.5 * math.sqrt((s - 1) * g * dd**3) if th > tc else 0.0
        return th, tc, ds, q

    res = _map(lambda *a: one(*a), tau, d)
    th, tc, ds, q = res if isinstance(res, tuple) else (list(z) for z in zip(*res))
    return RichResult(payload={"theta": th, "theta_cr": tc, "d_star": ds, "bedload": q})


def oil_spill_drift(x0, y0, current, wind, hours, dt=1.0, n_particles=100, wind_factor=0.03, diffusivity=10.0, seed=0):
    r"""Lagrangian drift of an oil slick: advection by current plus a wind-drift fraction, and diffusion.

    Each particle moves ``dx = (U_c + alpha U_w) dt + sqrt(2 K dt) xi`` per
    step (``alpha`` wind drift factor, 3 percent by default; ``K``
    horizontal diffusivity m^2/s; ``xi`` standard normal from the Philox
    stream), with ``current`` and ``wind`` as constant ``(u, v)`` in m/s,
    positions in m and ``dt`` in hours.

    References
    ----------
    ASCE Task Committee (1996). State-of-the-art review of modeling transport
    and fate of oil spills. *Journal of Hydraulic Engineering* 122, 594-609.

    Examples
    --------
    >>> r = oil_spill_drift(0.0, 0.0, (0.1, 0.0), (5.0, 0.0), 10, n_particles=4, diffusivity=0.0)
    >>> [round(v, 6) for v in r.centroid]
    [9000.0, 0.0]
    """
    steps = int(round(hours / dt))
    sec = dt * 3600.0
    ux = current[0] + wind_factor * wind[0]
    uy = current[1] + wind_factor * wind[1]
    sd = math.sqrt(2.0 * diffusivity * sec)
    z = [float(v) for v in random_normal(2 * steps * n_particles, seed=seed)]
    xs = [float(x0)] * n_particles
    ys = [float(y0)] * n_particles
    k = 0
    for _ in range(steps):
        for p in range(n_particles):
            xs[p] += ux * sec + sd * z[k]
            ys[p] += uy * sec + sd * z[k + 1]
            k += 2
    return RichResult(payload={"x": xs, "y": ys, "centroid": [ssum(xs) / n_particles, ssum(ys) / n_particles]})


def plume_concentration(x, y, load, u, depth, ky, decay=0.0):
    r"""Depth-averaged concentration of a continuous point discharge in a uniform coastal current.

    ``C(x, y) = M / (h sqrt(4 pi K_y x u)) exp(-u y^2 / (4 K_y x)) exp(-k x / u)``
    for ``x > 0`` downstream (0 upstream): load ``M`` (mass/s), current ``u``
    (m/s), depth ``h`` (m), transverse diffusivity ``K_y`` (m^2/s) and
    first-order decay ``k`` (1/s).

    References
    ----------
    Fischer, H. B., List, E. J., Koh, R. C. Y., Imberger, J. and Brooks, N. H.
    (1979). *Mixing in Inland and Coastal Waters*. Academic Press, chapter 5.

    Examples
    --------
    >>> round(plume_concentration(1000.0, 20.0, 5.0, 0.2, 10.0, 1.0), 12)
    0.009776067349
    """

    def one(xx, yy):
        if xx <= 0:
            return 0.0
        return (
            load
            / (depth * math.sqrt(4 * math.pi * ky * xx * u))
            * math.exp(-u * yy * yy / (4 * ky * xx))
            * math.exp(-decay * xx / u)
        )

    return _map(one, x, y)


def depth_invariant_index(band_i, band_j, deep_i, deep_j, sand_i, sand_j):
    r"""Lyzenga's depth-invariant bottom index for benthic habitat (coral, seagrass) mapping.

    ``X_k = ln(L_k - L_sk)`` after removing the deep-water signal ``L_sk``; the
    attenuation ratio ``k_i / k_j = a + sqrt(a^2 + 1)`` with ``a = (var X_i -
    var X_j) / (2 cov(X_i, X_j))`` from pixels of a uniform bottom (sand) at
    varying depth; ``DII = X_i - (k_i / k_j) X_j``.

    References
    ----------
    Lyzenga, D. R. (1981). Remote sensing of bottom reflectance and water
    attenuation parameters in shallow water using aircraft and Landsat data.
    *International Journal of Remote Sensing* 2, 71-82.

    Examples
    --------
    >>> r = depth_invariant_index([0.08, 0.06], [0.05, 0.035], 0.01, 0.005, [0.09, 0.07, 0.05], [0.06, 0.042, 0.027])
    >>> round(r.ratio, 10)
    0.7577713816
    """
    si = [math.log(v - deep_i) for v in _vec(sand_i)]
    sj = [math.log(v - deep_j) for v in _vec(sand_j)]
    n = len(si)
    mi, mj = ssum(si) / n, ssum(sj) / n
    vi = ssum((v - mi) ** 2 for v in si) / (n - 1)
    vj = ssum((v - mj) ** 2 for v in sj) / (n - 1)
    cij = ssum((a - mi) * (b - mj) for a, b in zip(si, sj)) / (n - 1)
    a = (vi - vj) / (2 * cij)
    ratio = a + math.sqrt(a * a + 1)
    dii = [math.log(x - deep_i) - ratio * math.log(y - deep_j) for x, y in zip(_vec(band_i), _vec(band_j))]
    return RichResult(payload={"dii": dii, "ratio": ratio})


def floating_algae_index(red, nir, swir, wavelengths=(645.0, 859.0, 1240.0)):
    r"""Floating algae index (Hu 2009) for kelp and Sargassum mapping.

    ``FAI = R_NIR - R'_NIR``, ``R'_NIR = R_red + (R_SWIR - R_red) (l_NIR -
    l_red) / (l_SWIR - l_red)`` (MODIS 645, 859 and 1240 nm by default).

    References
    ----------
    Hu, C. (2009). A novel ocean color index to detect floating algae in the
    global oceans. *Remote Sensing of Environment* 113, 2118-2129.

    Examples
    --------
    >>> round(floating_algae_index(0.04, 0.12, 0.05), 10)
    0.0764033613
    """
    lr, ln, ls = [float(v) for v in wavelengths]
    return _map(lambda r, n, s: n - (r + (s - r) * (ln - lr) / (ls - lr)), red, nir, swir)


def mangrove_vegetation_index(green, nir, swir1):
    r"""Mangrove vegetation index ``MVI = (NIR - Green) / (SWIR1 - Green)`` (Baloloy et al. 2020).

    References
    ----------
    Baloloy, A. B. et al. (2020). Development and application of a new mangrove
    vegetation index (MVI) for rapid and accurate mangrove mapping. *ISPRS
    Journal of Photogrammetry and Remote Sensing* 166, 95-117.

    Examples
    --------
    >>> round(mangrove_vegetation_index(0.05, 0.3, 0.12), 10)
    3.5714285714
    """
    return _map(lambda g, n, s: (n - g) / (s - g), green, nir, swir1)


def bare_soil_index(blue, red, nir, swir):
    r"""Bare soil index ``BSI = ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))`` (Rikimaru et al. 2002).

    References
    ----------
    Rikimaru, A., Roy, P. S. and Miyatake, S. (2002). Tropical forest cover
    density mapping. *Tropical Ecology* 43, 39-47.

    Examples
    --------
    >>> round(bare_soil_index(0.08, 0.2, 0.25, 0.3), 10)
    0.2048192771
    """
    return _map(lambda b, r, n, s: ((s + r) - (n + b)) / ((s + r) + (n + b)), blue, red, nir, swir)


def cheatsheet() -> str:
    return (
        "ocean_chlorophyll / vgpm_production / euphotic_depth / co2_flux / oxygen_solubility / "
        "practical_salinity / split_window_sst / sediment_transport / oil_spill_drift / "
        "plume_concentration / depth_invariant_index / floating_algae_index / "
        "mangrove_vegetation_index / bare_soil_index -> marine."
    )
