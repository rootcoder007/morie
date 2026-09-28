# morie.fn -- function file (rootcoder007/morie)
"""Map projections (Snyder 1987; Karney 2011), WGS84 local tangent planes and ellipsoidal geodesics (Vincenty 1975)."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["map_project", "utm_zone", "geodetic_to_enu", "great_circle_distance", "vincenty_inverse"]

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
_E2 = WGS84_F * (2.0 - WGS84_F)
_E = math.sqrt(_E2)


def _q(phi):
    s = math.sin(phi)
    return (1 - _E2) * (s / (1 - _E2 * s * s) - math.log((1 - _E * s) / (1 + _E * s)) / (2 * _E))


def _m(phi):
    return math.cos(phi) / math.sqrt(1 - _E2 * math.sin(phi) ** 2)


def _t(phi):
    s = math.sin(phi)
    return math.tan(math.pi / 4 - phi / 2) / ((1 - _E * s) / (1 + _E * s)) ** (_E / 2)


def _newton(f, df, x0):
    x = x0
    for _ in range(100):
        dx = f(x) / df(x)
        x -= dx
        if abs(dx) < 1e-15:
            break
    return x


def _tmerc(lam, phi, lon0, k0, fe, fn):
    """Ellipsoidal transverse Mercator by the Krueger n-series to order n^6 (Karney 2011)."""
    n = WGS84_F / (2 - WGS84_F)
    A = WGS84_A / (1 + n) * (1 + n**2 / 4 + n**4 / 64 + n**6 / 256)
    al = [
        n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16 + 41 * n**4 / 180 - 127 * n**5 / 288 + 7891 * n**6 / 37800,
        13 * n**2 / 48 - 3 * n**3 / 5 + 557 * n**4 / 1440 + 281 * n**5 / 630 - 1983433 * n**6 / 1935360,
        61 * n**3 / 240 - 103 * n**4 / 140 + 15061 * n**5 / 26880 + 167603 * n**6 / 181440,
        49561 * n**4 / 161280 - 179 * n**5 / 168 + 6601661 * n**6 / 7257600,
        34729 * n**5 / 80640 - 3418889 * n**6 / 1995840,
        212378941 * n**6 / 319334400,
    ]
    dl = lam - lon0
    t = math.sinh(math.atanh(math.sin(phi)) - _E * math.atanh(_E * math.sin(phi)))
    xi = math.atan2(t, math.cos(dl))
    eta = math.atanh(math.sin(dl) / math.sqrt(1 + t * t))
    x = eta + sum(al[j - 1] * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j in range(1, 7))
    y = xi + sum(al[j - 1] * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j in range(1, 7))
    return fe + k0 * A * x, fn + k0 * A * y


def map_project(
    lon,
    lat,
    proj: str,
    *,
    R: float = 6371000.0,
    lon_0: float = 0.0,
    lat_0: float = 0.0,
    lat_1: float | None = None,
    lat_2: float | None = None,
    k_0: float = 1.0,
    zone: int | None = None,
    south: bool = False,
) -> RichResult:
    r"""Forward map projection of geographic coordinates (degrees) to planar coordinates (metres).

    Spherical (radius ``R``, as PROJ with ``+R``; Snyder 1987): ``sinu``
    (sinusoidal), ``moll`` (Mollweide, Newton on ``2t + sin 2t = pi sin
    phi``), ``eck4`` (Eckert IV), ``goode`` (Goode homolosine: sinusoidal
    within 40d44'11.8'' of the equator, Mollweide shifted by ``0.05280 R``
    beyond, PROJ's constants), ``wintri`` (Winkel tripel, ``lat_1`` default
    Winkel's ``acos(2/pi)``), ``bonne`` (standard
    parallel ``lat_1``), ``cass`` (Cassini), ``aeqd`` (azimuthal
    equidistant about ``(lon_0, lat_0)``), ``stere`` (oblique stereographic,
    scale ``k_0``).  Ellipsoidal WGS84 (Snyder 1987, chapters 7, 14, 15):
    ``merc`` (Mercator, ``k_0``), ``lcc`` (Lambert conformal conic, two
    standard parallels), ``aea`` (Albers equal-area conic), ``tmerc``
    (transverse Mercator, Krueger series to ``n^6``; Karney 2011) and
    ``utm`` (zone ``zone``, ``k_0 = 0.9996``, false easting 500 km, false
    northing 10 000 km in the south).  Longitude differences are wrapped to
    ``[-180, 180)`` degrees.

    References
    ----------
    Snyder, J. P. (1987). *Map Projections: A Working Manual*. U.S.
    Geological Survey Professional Paper 1395.
    Karney, C. F. F. (2011). Transverse Mercator with an accuracy of a few
    nanometers. *Journal of Geodesy*, 85(8), 475-485.

    Examples
    --------
    >>> [round(v, 3) for v in map_project(10.0, 50.0, "sinu").xy]
    [714747.211, 5559746.332]
    """
    lam, phi = math.radians(lon), math.radians(lat)
    l0, p0 = math.radians(lon_0), math.radians(lat_0)
    dl = (lam - l0 + math.pi) % (2 * math.pi) - math.pi  # longitude difference wrapped to [-pi, pi), as PROJ
    if proj == "sinu":
        x, y = R * dl * math.cos(phi), R * phi
    elif proj in ("moll", "goode"):
        lim = math.radians(40 + 44 / 60 + 11.8 / 3600)
        if proj == "goode" and abs(phi) <= lim:
            x, y = R * dl * math.cos(phi), R * phi
        else:
            if abs(abs(phi) - math.pi / 2) < 1e-15:
                th = math.copysign(math.pi / 2, phi)
            else:
                th = _newton(
                    lambda t: 2 * t + math.sin(2 * t) - math.pi * math.sin(phi), lambda t: 2 + 2 * math.cos(2 * t), phi
                )
            x = 2 * math.sqrt(2) / math.pi * R * dl * math.cos(th)
            y = math.sqrt(2) * R * math.sin(th)
            if proj == "goode":
                y -= math.copysign(0.05280 * R, phi)  # PROJ's Y_COR
    elif proj == "eck4":
        c = 2 + math.pi / 2
        if abs(abs(phi) - math.pi / 2) < 1e-15:
            th = math.copysign(math.pi / 2, phi)
        else:
            th = _newton(
                lambda t: t + math.sin(t) * math.cos(t) + 2 * math.sin(t) - c * math.sin(phi),
                lambda t: 2 * math.cos(t) * (1 + math.cos(t)),
                phi / 2,
            )
        x = 2 / math.sqrt(math.pi * (4 + math.pi)) * R * dl * (1 + math.cos(th))
        y = 2 * math.sqrt(math.pi / (4 + math.pi)) * R * math.sin(th)
    elif proj == "wintri":
        p1 = math.acos(2 / math.pi) if lat_1 is None else math.radians(lat_1)
        a = math.acos(math.cos(phi) * math.cos(dl / 2))
        sinc = math.sin(a) / a if a != 0 else 1.0
        x = 0.5 * R * (dl * math.cos(p1) + 2 * math.cos(phi) * math.sin(dl / 2) / sinc)
        y = 0.5 * R * (phi + math.sin(phi) / sinc)
    elif proj == "bonne":
        p1 = math.radians(45.0 if lat_1 is None else lat_1)
        rho = 1 / math.tan(p1) + p1 - phi
        E = dl * math.cos(phi) / rho if rho != 0 else 0.0
        x, y = R * rho * math.sin(E), R * (1 / math.tan(p1) - rho * math.cos(E))
    elif proj == "cass":
        x = R * math.asin(math.cos(phi) * math.sin(dl))
        y = R * (math.atan2(math.tan(phi), math.cos(dl)) - p0)
    elif proj in ("aeqd", "stere"):
        cc = math.sin(p0) * math.sin(phi) + math.cos(p0) * math.cos(phi) * math.cos(dl)
        if proj == "aeqd":
            c = math.acos(max(-1.0, min(1.0, cc)))
            k = c / math.sin(c) if c != 0 else 1.0
        else:
            k = 2 * k_0 / (1 + cc)
        x = R * k * math.cos(phi) * math.sin(dl)
        y = R * k * (math.cos(p0) * math.sin(phi) - math.sin(p0) * math.cos(phi) * math.cos(dl))
    elif proj == "merc":
        s = math.sin(phi)
        x = WGS84_A * k_0 * dl
        y = WGS84_A * k_0 * math.log(math.tan(math.pi / 4 + phi / 2) * ((1 - _E * s) / (1 + _E * s)) ** (_E / 2))
    elif proj == "lcc":
        p1, p2 = math.radians(lat_1), math.radians(lat_2 if lat_2 is not None else lat_1)
        n = (math.log(_m(p1)) - math.log(_m(p2))) / (math.log(_t(p1)) - math.log(_t(p2))) if p1 != p2 else math.sin(p1)
        F = _m(p1) / (n * _t(p1) ** n)
        rho, rho0 = WGS84_A * F * _t(phi) ** n, WGS84_A * F * _t(p0) ** n
        x, y = rho * math.sin(n * dl), rho0 - rho * math.cos(n * dl)
    elif proj == "aea":
        p1, p2 = math.radians(lat_1), math.radians(lat_2 if lat_2 is not None else lat_1)
        m1, m2 = _m(p1), _m(p2)
        q1, q2 = _q(p1), _q(p2)
        n = (m1 * m1 - m2 * m2) / (q2 - q1) if p1 != p2 else math.sin(p1)
        C = m1 * m1 + n * q1
        rho = WGS84_A * math.sqrt(C - n * _q(phi)) / n
        rho0 = WGS84_A * math.sqrt(C - n * _q(p0)) / n
        x, y = rho * math.sin(n * dl), rho0 - rho * math.cos(n * dl)
    elif proj == "tmerc":
        x, y = _tmerc(lam, phi, l0, k_0, 0.0, 0.0)
    elif proj == "utm":
        z = utm_zone(lon, lat) if zone is None else int(zone)
        x, y = _tmerc(lam, phi, math.radians(-183.0 + 6.0 * z), 0.9996, 500000.0, 10000000.0 if south else 0.0)
    else:
        raise ValueError("unknown projection")
    return RichResult(payload={"xy": [x, y], "proj": proj})


def utm_zone(lon: float, lat: float) -> int:
    r"""UTM zone of a WGS84 point, with the Norway (32V) and Svalbard (31X-37X) exceptions.

    Examples
    --------
    >>> utm_zone(10.0, 50.0), utm_zone(5.0, 60.0), utm_zone(10.0, 75.0)
    (32, 32, 33)
    """
    z = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        return 32
    if 72.0 <= lat < 84.0 and lon >= 0.0:
        if lon < 9.0:
            return 31
        if lon < 21.0:
            return 33
        if lon < 33.0:
            return 35
        if lon < 42.0:
            return 37
    return z


def _ecef(lon, lat, h):
    lam, phi = math.radians(lon), math.radians(lat)
    N = WGS84_A / math.sqrt(1 - _E2 * math.sin(phi) ** 2)
    return (
        (N + h) * math.cos(phi) * math.cos(lam),
        (N + h) * math.cos(phi) * math.sin(lam),
        (N * (1 - _E2) + h) * math.sin(phi),
    )


def geodetic_to_enu(lon, lat, h, lon0, lat0, h0) -> RichResult:
    r"""WGS84 geodetic to local east-north-up coordinates about ``(lon0, lat0, h0)``.

    Earth-centred Earth-fixed coordinates ``((N + h) cos phi cos lambda, (N +
    h) cos phi sin lambda, (N(1 - e^2) + h) sin phi)`` relative to the origin,
    rotated by ``e = (-sin l0, cos l0, 0)``, ``n = (-sin p0 cos l0, -sin p0
    sin l0, cos p0)``, ``u = (cos p0 cos l0, cos p0 sin l0, sin p0)`` (as PROJ
    ``cart`` + ``topocentric``).

    Examples
    --------
    >>> [round(v, 6) for v in geodetic_to_enu(0.0, 0.0, 10.0, 0.0, 0.0, 0.0).enu]
    [0.0, 0.0, 10.0]
    """
    X, Y, Z = _ecef(lon, lat, h)
    X0, Y0, Z0 = _ecef(lon0, lat0, h0)
    dx, dy, dz = X - X0, Y - Y0, Z - Z0
    l0, p0 = math.radians(lon0), math.radians(lat0)
    e = -math.sin(l0) * dx + math.cos(l0) * dy
    nn = -math.sin(p0) * math.cos(l0) * dx - math.sin(p0) * math.sin(l0) * dy + math.cos(p0) * dz
    u = math.cos(p0) * math.cos(l0) * dx + math.cos(p0) * math.sin(l0) * dy + math.sin(p0) * dz
    return RichResult(payload={"enu": [e, nn, u], "ecef": [X, Y, Z]})


def great_circle_distance(lon1, lat1, lon2, lat2, *, r: float = 6378137.0) -> float:
    r"""Great-circle distance by the haversine formula (Sinnott 1984), as ``geosphere::distHaversine``.

    Examples
    --------
    >>> round(great_circle_distance(0.0, 0.0, 90.0, 0.0, r=1.0), 12)
    1.570796326795
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def vincenty_inverse(lon1, lat1, lon2, lat2, *, tol: float = 1e-13, maxit: int = 200) -> RichResult:
    r"""Ellipsoidal (WGS84) distance and azimuths by Vincenty's (1975) inverse formula.

    Iterates the longitude on the auxiliary sphere until it changes less
    than ``tol``; returns the geodesic ``distance`` (m) and the forward and
    back azimuths in degrees clockwise from north (``nan`` for nearly
    antipodal points that do not converge).

    References
    ----------
    Vincenty, T. (1975). Direct and inverse solutions of geodesics on the
    ellipsoid with application of nested equations. *Survey Review*,
    23(176), 88-93.

    Examples
    --------
    >>> round(vincenty_inverse(0.0, 0.0, 1.0, 0.0).distance, 6)
    111319.490793
    """
    a, f = WGS84_A, WGS84_F
    b = a * (1 - f)
    L = math.radians(lon2 - lon1)
    U1 = math.atan((1 - f) * math.tan(math.radians(lat1)))
    U2 = math.atan((1 - f) * math.tan(math.radians(lat2)))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    conv = False
    for _ in range(maxit):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.sqrt((cU2 * sl) ** 2 + (cU1 * sU2 - sU1 * cU2 * cl) ** 2)
        if ss == 0:
            return RichResult(payload={"distance": 0.0, "azimuth1": 0.0, "azimuth2": 0.0})
        cs = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(ss, cs)
        sa = cU1 * cU2 * sl / ss
        c2a = 1 - sa * sa
        c2sm = cs - 2 * sU1 * sU2 / c2a if c2a != 0 else 0.0
        C = f / 16 * c2a * (4 + f * (4 - 3 * c2a))
        lp = lam
        lam = L + (1 - C) * f * sa * (sig + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm * c2sm)))
        if abs(lam - lp) < tol:
            conv = True
            break
    if not conv:
        nan = float("nan")
        return RichResult(payload={"distance": nan, "azimuth1": nan, "azimuth2": nan})
    u2 = c2a * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds = (
        B
        * ss
        * (c2sm + B / 4 * (cs * (-1 + 2 * c2sm * c2sm) - B / 6 * c2sm * (-3 + 4 * ss * ss) * (-3 + 4 * c2sm * c2sm)))
    )
    s = b * A * (sig - ds)
    az1 = math.degrees(math.atan2(cU2 * math.sin(lam), cU1 * sU2 - sU1 * cU2 * math.cos(lam)))
    az2 = math.degrees(math.atan2(cU1 * math.sin(lam), -sU1 * cU2 + cU1 * sU2 * math.cos(lam)))
    return RichResult(payload={"distance": s, "azimuth1": az1 % 360.0, "azimuth2": az2 % 360.0})


def cheatsheet() -> str:
    return "project / utm_zone / geodetic_to_enu / vincenty_inverse -> map projections and geodesy."
