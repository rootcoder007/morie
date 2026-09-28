# morie.fn -- function file (rootcoder007/morie)
"""Further map projections and geodesy: Robinson (PROJ's tables), Web Mercator, inverse projections,
rotated-pole grids (PROJ ob_tran) and geoid heights by spherical-harmonic synthesis."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = [
    "robinson_project",
    "web_mercator",
    "map_unproject",
    "rotated_pole",
    "rotated_grid",
    "normal_gravity",
    "geoid_height",
]

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_E = math.sqrt(_E2)

# PROJ robin.cpp coefficient tables (float32 literals, 5-degree nodes)
_ROBX = [
    (1.0, 2.21989997769713e-17, -7.155149796744809e-05, 3.1102999855647795e-06),
    (0.9986000061035156, -0.0004822429909836501, -2.4896999093471095e-05, -1.3308999768923968e-06),
    (0.9954000115394592, -0.0008310300181619823, -4.486049874685705e-05, -9.867010248854058e-07),
    (0.9900000095367432, -0.0013536399928852916, -5.966100070509128e-05, 3.677700078696944e-06),
    (0.982200026512146, -0.001674419967457652, -4.495469966059318e-06, -5.724109996663174e-06),
    (0.9729999899864197, -0.0021486799232661724, -9.035709808813408e-05, 1.8735999418595384e-08),
    (0.9599999785423279, -0.0030508500058203936, -9.007610060507432e-05, 1.6491700307597057e-06),
    (0.9427000284194946, -0.003827919950708747, -6.533860141644254e-05, -2.6154000352107687e-06),
    (0.9215999841690063, -0.004677460063248873, -0.00010456999734742567, 4.812429779121885e-06),
    (0.8962000012397766, -0.005362229887396097, -3.2383100915467367e-05, -5.43431997357402e-06),
    (0.867900013923645, -0.006093630101531744, -0.00011389800056349486, 3.324840008644969e-06),
    (0.8349999785423279, -0.006983249913901091, -6.402529834304005e-05, 9.34959018650261e-07),
    (0.7986000180244446, -0.007553379982709885, -5.000090095563792e-05, 9.353240102427662e-07),
    (0.7597000002861023, -0.00798324029892683, -3.5970999306300655e-05, -2.276259920108714e-06),
    (0.7185999751091003, -0.008513670414686203, -7.011489651631564e-05, -8.63029981701402e-06),
    (0.6732000112533569, -0.009862090460956097, -0.00019956899632234126, 1.919739952427335e-05),
    (0.6212999820709229, -0.0104179996997118, 8.839229849399999e-05, 6.240510174393421e-06),
    (0.5722000002861023, -0.009066009894013405, 0.00018200000340584666, 6.240510174393421e-06),
    (0.5321999788284302, -0.006777970120310783, 0.0002756080066319555, 6.240510174393421e-06),
]
_ROBY = [
    (-5.204170014340115e-18, 0.012400000356137753, 1.2143100314194296e-18, -8.452839816985858e-11),
    (0.06199999898672104, 0.012400000356137753, -1.267929983228555e-09, 4.226420047270807e-10),
    (0.12399999797344208, 0.012400000356137753, 5.071710162951604e-09, -1.6060399676831594e-09),
    (0.1860000044107437, 0.012399899773299694, -1.9018900232481428e-08, 6.001520169718333e-09),
    (0.24799999594688416, 0.01240019965916872, 7.100390320147199e-08, -2.240000007702747e-08),
    (0.3100000023841858, 0.012399200350046158, -2.6499699856685766e-07, 8.359860004247821e-08),
    (0.3720000088214874, 0.01240289956331253, 9.88982947092154e-07, -3.119940004125965e-07),
    (0.4339999854564667, 0.012389300391077995, -3.6909300433762837e-06, -4.3562098994698317e-07),
    (0.4957999885082245, 0.012319800443947315, -1.0225199730484746e-05, -3.455230057625158e-07),
    (0.5570999979972839, 0.012191600166261196, -1.540810080769006e-05, -5.822880098094174e-07),
    (0.6176000237464905, 0.011993800289928913, -2.4142400434357114e-05, -5.253269819149864e-07),
    (0.6769000291824341, 0.011713000014424324, -3.202230072929524e-05, -5.164050094208505e-07),
    (0.7346000075340271, 0.011354099959135056, -3.976840162067674e-05, -6.090519946155837e-07),
    (0.7903000116348267, 0.01091070007532835, -4.8904199502430856e-05, -1.0473900147189852e-06),
    (0.843500018119812, 0.010343099944293499, -6.461500015575439e-05, -1.4037400131172717e-09),
    (0.8935999870300293, 0.009696859866380692, -6.463599856942892e-05, -8.54700010677334e-06),
    (0.9394000172615051, 0.008409470319747925, -0.00019284100562799722, -4.210599854559405e-06),
    (0.9761000275611877, 0.0061652702279388905, -0.00025599999935366213, -4.210599854559405e-06),
    (1.0, 0.0032894699834287167, -0.0003191590076312423, -4.210599854559405e-06),
]


def _v(c, z):
    return c[0] + z * (c[1] + z * (c[2] + z * c[3]))


def _dv(c, z):
    return c[1] + 2 * z * c[2] + z * z * 3.0 * c[3]


def _wrap(dl):
    return (dl + math.pi) % (2 * math.pi) - math.pi


def robinson_project(lon: float, lat: float, *, R: float = _A, lon_0: float = 0.0) -> list:
    r"""Robinson projection (degrees to metres) by PROJ's cubic interpolation of Robinson's 5-degree table.

    ``x = 0.8487 R X(phi) lambda``, ``y = 1.3523 R Y(phi)`` with ``X``, ``Y``
    cubic in the offset (degrees) from the tabulated node below ``|phi|``,
    using PROJ's coefficients so that results equal ``+proj=robin``.

    References
    ----------
    Robinson, A. H. (1974). A new map projection: its development and
    characteristics. *International Yearbook of Cartography*, 14, 145-155.
    Snyder, J. P. (1990). The Robinson projection: a computation algorithm.
    *Cartography and Geographic Information Systems*, 17, 301-305.

    Examples
    --------
    >>> [round(v, 3) for v in robinson_project(10.0, 50.0)]
    [819964.61, 5326895.726]
    """
    phi = math.radians(lat)
    lam = _wrap(math.radians(lon - lon_0))
    d = abs(phi)
    i = min(int(math.floor(d * 11.45915590261646417544 + 1e-15)), 18)
    dd = math.degrees(d - 0.08726646259971647884 * i)
    x = _v(_ROBX[i], dd) * 0.8487 * lam * R
    y = _v(_ROBY[i], dd) * 1.3523 * R
    return [x, -y if phi < 0 else y]


def web_mercator(lon: float, lat: float) -> list:
    r"""Web (pseudo-)Mercator EPSG:3857: the spherical Mercator formulas on the WGS84 semi-major axis.

    ``x = a lambda``, ``y = a ln tan(pi/4 + phi/2)``; not conformal on the ellipsoid.

    References
    ----------
    NGA (2014). *Implementation Practice: Web Mercator Map Projection*. NGA.SIG.0011_1.0.0_WEBMERC.

    Examples
    --------
    >>> [round(v, 3) for v in web_mercator(10.0, 50.0)]
    [1113194.908, 6446275.841]
    """
    return [_A * math.radians(lon), _A * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))]


def map_unproject(
    x: float, y: float, proj: str, *, R: float | None = None, lon_0: float = 0.0, k_0: float = 1.0
) -> list:
    r"""Inverse map projection, planar metres to ``[lon, lat]`` degrees.

    ``"webmerc"`` (EPSG:3857: ``phi = 2 atan(e^{y/a}) - pi/2``), ``"merc"``
    (ellipsoidal WGS84 Mercator, fixed-point iteration of Snyder eq. 7-9),
    ``"sinu"``, ``"moll"`` (spherical, radius ``R`` default 6371000 as
    :func:`morie.fn.mapproj.map_project`) and ``"robin"`` (PROJ's Newton
    inversion of the Robinson table, ``R`` default ``a``).

    References
    ----------
    Snyder, J. P. (1987). *Map Projections: A Working Manual*. USGS
    Professional Paper 1395, pp. 44, 243, 252.

    Examples
    --------
    >>> [round(v, 9) for v in map_unproject(*web_mercator(10.0, 50.0), "webmerc")]
    [10.0, 50.0]
    """
    l0 = math.radians(lon_0)
    if proj == "webmerc":
        lam, phi = x / _A, 2 * math.atan(math.exp(y / _A)) - math.pi / 2
    elif proj == "merc":
        t = math.exp(-y / (_A * k_0))
        phi = math.pi / 2 - 2 * math.atan(t)
        for _ in range(100):
            s = _E * math.sin(phi)
            new = math.pi / 2 - 2 * math.atan(t * ((1 - s) / (1 + s)) ** (_E / 2))
            if abs(new - phi) < 1e-15:
                phi = new
                break
            phi = new
        lam = x / (_A * k_0)
    elif proj == "sinu":
        Rr = 6371000.0 if R is None else R
        phi = y / Rr
        lam = x / (Rr * math.cos(phi))
    elif proj == "moll":
        Rr = 6371000.0 if R is None else R
        th = math.asin(y / (math.sqrt(2) * Rr))
        phi = math.asin((2 * th + math.sin(2 * th)) / math.pi)
        lam = math.pi * x / (2 * math.sqrt(2) * Rr * math.cos(th))
    elif proj == "robin":
        Rr = _A if R is None else R
        lam = x / Rr / 0.8487
        p = abs(y / Rr / 1.3523)
        if p >= 1.0:
            phi = math.copysign(math.pi / 2, y)
            lam /= _ROBX[18][0]
        else:
            i = int(math.floor(p * 18))
            while True:
                if _ROBY[i][0] > p:
                    i -= 1
                elif _ROBY[i + 1][0] <= p:
                    i += 1
                else:
                    break
            T = _ROBY[i]
            t = 5.0 * (p - T[0]) / (_ROBY[i + 1][0] - T[0])
            for _ in range(100):
                t1 = (_v(T, t) - p) / _dv(T, t)
                t -= t1
                if abs(t1) < 1e-10:
                    break
            phi = math.radians(5 * i + t)
            if y < 0:
                phi = -phi
            lam /= _v(_ROBX[i], t)
    else:
        raise ValueError("proj must be webmerc, merc, sinu, moll or robin")
    return [math.degrees(_wrap(lam + l0)), math.degrees(phi)]


def rotated_pole(lon: float, lat: float, pole_lon: float, pole_lat: float, *, inverse: bool = False) -> list:
    r"""Geographic to rotated-pole coordinates (``inverse=True``: back), as PROJ ``ob_tran`` / CF ``rotated_latitude_longitude``.

    The rotated grid has its north pole at ``(pole_lon, pole_lat)`` in true
    coordinates (CF ``grid_north_pole_longitude/latitude``), i.e. PROJ
    ``+proj=ob_tran +o_proj=longlat +o_lat_p=pole_lat +o_lon_p=0 +lon_0=180+pole_lon``:
    forward ``lambda' = atan2(cos phi sin lambda, sin phi_p cos phi cos lambda + cos phi_p sin phi)``,
    ``phi' = asin(sin phi_p sin phi - cos phi_p cos phi cos lambda)`` with ``lambda`` measured from ``lon_0``.

    References
    ----------
    Snyder, J. P. (1987). *Map Projections: A Working Manual*, pp. 29-32 (oblique transformation).
    Eaton, B. et al. (2023). *NetCDF Climate and Forecast (CF) Metadata Conventions*, section 5.6.

    Examples
    --------
    >>> [round(v, 9) for v in rotated_pole(10.0, 50.0, 10.0, 40.0)]
    [0.0, 80.0]
    """
    pp = math.radians(pole_lat)
    sp, cp = math.sin(pp), math.cos(pp)
    lon0 = math.radians(180.0 + pole_lon)
    phi = math.radians(lat)
    if not inverse:
        lam = _wrap(math.radians(lon) - lon0)
        cl, cf, sf = math.cos(lam), math.cos(phi), math.sin(phi)
        lo = math.atan2(cf * math.sin(lam), sp * cf * cl + cp * sf)
        la = math.asin(max(-1.0, min(1.0, sp * sf - cp * cf * cl)))
        return [math.degrees(_wrap(lo)), math.degrees(la)]
    lam = math.radians(lon)
    cl, cf, sf = math.cos(lam), math.cos(phi), math.sin(phi)
    lo = math.atan2(cf * math.sin(lam), sp * cf * cl - cp * sf)
    la = math.asin(max(-1.0, min(1.0, sp * sf + cp * cf * cl)))
    return [math.degrees(_wrap(lo + lon0)), math.degrees(la)]


def rotated_grid(rlon, rlat, pole_lon: float, pole_lat: float) -> RichResult:
    r"""True longitudes and latitudes of a regular grid defined in rotated-pole coordinates.

    ``rlon`` and ``rlat`` are the rotated axes (degrees); every node
    ``(rlon_j, rlat_i)`` is mapped back by :func:`rotated_pole` (inverse).
    Returns ``lon`` and ``lat`` matrices (rows follow ``rlat``).

    Examples
    --------
    >>> g = rotated_grid([-1.0, 0.0, 1.0], [0.0], 180.0, 90.0)
    >>> [round(v, 9) for v in g.lon[0]], [round(v, 9) for v in g.lat[0]]
    ([-1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    """
    lon, lat = [], []
    for b in rlat:
        rowx, rowy = [], []
        for a in rlon:
            x, y = rotated_pole(float(a), float(b), pole_lon, pole_lat, inverse=True)
            rowx.append(x)
            rowy.append(y)
        lon.append(rowx)
        lat.append(rowy)
    return RichResult(payload={"lon": lon, "lat": lat})


def normal_gravity(lat: float) -> float:
    r"""WGS84 normal gravity on the ellipsoid by Somigliana's closed formula (m s^-2).

    ``gamma = gamma_e (1 + k sin^2 phi) / sqrt(1 - e^2 sin^2 phi)`` with
    ``gamma_e = 9.7803253359`` and ``k = 0.00193185265241``.

    References
    ----------
    NIMA (2000). *Department of Defense World Geodetic System 1984*. TR8350.2, 3rd edn, eq. 4-1.

    Examples
    --------
    >>> round(normal_gravity(45.0), 10)
    9.8061977694
    """
    s2 = math.sin(math.radians(lat)) ** 2
    return 9.7803253359 * (1 + 0.00193185265241 * s2) / math.sqrt(1 - _E2 * s2)


def geoid_height(lon: float, lat: float, C, S, *, gm: float = 3.986004418e14, a: float = _A) -> float:
    r"""Geoid undulation by Bruns' formula from fully normalised disturbing-potential coefficients.

    ``N = GM / (a gamma) sum_{n} sum_{m=0}^{n} (C_nm cos m lambda + S_nm sin m lambda) P_nm(sin phi)``
    (spherical approximation ``r = a``) with fully normalised associated
    Legendre functions by the standard column recursion and ``gamma`` from
    :func:`normal_gravity`. ``C[n][m]``, ``S[n][m]`` are the coefficients of the
    disturbing potential (the geopotential model minus the normal field),
    as published with EGM96/EGM2008 after subtracting the even zonal normal terms.

    References
    ----------
    Heiskanen, W. A. and Moritz, H. (1967). *Physical Geodesy*. Freeman, eq. 2-144 and 2-237.
    Holmes, S. A. and Featherstone, W. E. (2002). A unified approach to the Clenshaw summation and the
    recursive computation of very high degree and order normalised associated Legendre functions.
    *Journal of Geodesy*, 76, 279-299.

    Examples
    --------
    >>> C = [[0.0], [0.0, 0.0], [0.0, 0.0, 2.43e-6]]
    >>> S = [[0.0], [0.0, 0.0], [0.0, 0.0, -1.40e-6]]
    >>> round(geoid_height(30.0, 10.0, C, S), 6)
    0.03077
    """
    phi, lam = math.radians(lat), math.radians(lon)
    t, u = math.sin(phi), math.cos(phi)
    nmax = len(C) - 1
    P = [[0.0] * (n + 1) for n in range(nmax + 1)]
    P[0][0] = 1.0
    for m in range(1, nmax + 1):
        P[m][m] = (math.sqrt(3.0) if m == 1 else math.sqrt((2 * m + 1) / (2 * m))) * u * P[m - 1][m - 1]
    for m in range(0, nmax + 1):
        if m + 1 <= nmax:
            P[m + 1][m] = math.sqrt(2 * m + 3) * t * P[m][m]
        for n in range(m + 2, nmax + 1):
            anm = math.sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)))
            bnm = math.sqrt((2 * n + 1) * (n + m - 1) * (n - m - 1) / ((n - m) * (n + m) * (2 * n - 3)))
            P[n][m] = anm * t * P[n - 1][m] - bnm * P[n - 2][m]
    s = 0.0
    for n in range(nmax + 1):
        for m in range(min(n, len(C[n]) - 1) + 1):
            s += (C[n][m] * math.cos(m * lam) + S[n][m] * math.sin(m * lam)) * P[n][m]
    return gm / (a * normal_gravity(lat)) * s


def cheatsheet() -> str:
    return (
        "robinson_project / web_mercator / map_unproject / rotated_pole / rotated_grid / normal_gravity / "
        "geoid_height -> projections and geodesy."
    )
