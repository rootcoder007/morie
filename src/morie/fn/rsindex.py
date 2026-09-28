# morie.fn -- function file (rootcoder007/morie)
"""Spectral indices, reflectance and surface-temperature conversions, change vectors and map accuracy."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "spectral_index",
    "red_edge_position",
    "toa_reflectance",
    "fractional_vegetation_cover",
    "ndvi_emissivity",
    "land_surface_temperature",
    "shortwave_albedo",
    "change_vector",
    "accuracy_assessment",
]

_INDICES = (
    "NDVI",
    "EVI",
    "SAVI",
    "MSAVI",
    "GNDVI",
    "NDWI",
    "MNDWI",
    "NDBI",
    "NDMI",
    "NDSI",
    "NBR",
    "NBR2",
    "BAI",
    "MSR",
    "CIre",
    "NDRE",
    "PRI",
)


def _nd(a, b):
    return (a - b) / (a + b)


def _one(index, b):
    g = b.get
    if index == "NDVI":
        return _nd(g("nir"), g("red"))
    if index == "EVI":
        return 2.5 * (g("nir") - g("red")) / (g("nir") + 6.0 * g("red") - 7.5 * g("blue") + 1.0)
    if index == "SAVI":
        L = g("L", 0.5)
        return (1.0 + L) * (g("nir") - g("red")) / (g("nir") + g("red") + L)
    if index == "MSAVI":
        n = g("nir")
        return (2.0 * n + 1.0 - math.sqrt((2.0 * n + 1.0) ** 2 - 8.0 * (n - g("red")))) / 2.0
    if index == "GNDVI":
        return _nd(g("nir"), g("green"))
    if index == "NDWI":
        return _nd(g("green"), g("nir"))
    if index in ("MNDWI", "NDSI"):
        return _nd(g("green"), g("swir1"))
    if index == "NDBI":
        return _nd(g("swir1"), g("nir"))
    if index == "NDMI":
        return _nd(g("nir"), g("swir1"))
    if index == "NBR":
        return _nd(g("nir"), g("swir2"))
    if index == "NBR2":
        return _nd(g("swir1"), g("swir2"))
    if index == "BAI":
        return 1.0 / ((0.1 - g("red")) ** 2 + (0.06 - g("nir")) ** 2)
    if index == "MSR":
        sr = g("nir") / g("red")
        return (sr - 1.0) / math.sqrt(sr + 1.0)
    if index == "CIre":
        return g("nir") / g("rededge") - 1.0
    if index == "NDRE":
        return _nd(g("nir"), g("rededge"))
    if index == "PRI":
        return _nd(g("r531"), g("r570"))
    raise ValueError(f"index must be one of {_INDICES}")


def spectral_index(index: str, **bands):
    r"""Spectral index of reflectance bands (scalars or equal-length sequences).

    - ``NDVI`` ``(nir - red)/(nir + red)`` (Rouse et al. 1974);
    - ``EVI`` ``2.5 (nir - red)/(nir + 6 red - 7.5 blue + 1)`` (Huete et al. 2002);
    - ``SAVI`` ``(1 + L)(nir - red)/(nir + red + L)``, ``L = 0.5`` (Huete 1988);
    - ``MSAVI`` ``(2 nir + 1 - sqrt((2 nir + 1)^2 - 8 (nir - red)))/2`` (Qi et al. 1994);
    - ``GNDVI`` ``(nir - green)/(nir + green)`` (Gitelson et al. 1996);
    - ``NDWI`` ``(green - nir)/(green + nir)`` (McFeeters 1996);
    - ``MNDWI`` ``(green - swir1)/(green + swir1)`` (Xu 2006); ``NDSI`` the same
      band ratio for snow (Hall et al. 1995);
    - ``NDBI`` ``(swir1 - nir)/(swir1 + nir)`` (Zha et al. 2003);
    - ``NDMI`` ``(nir - swir1)/(nir + swir1)`` (Gao 1996);
    - ``NBR`` ``(nir - swir2)/(nir + swir2)`` and ``NBR2`` ``(swir1 - swir2)/(swir1 + swir2)``
      (Key and Benson 2006);
    - ``BAI`` ``1/((0.1 - red)^2 + (0.06 - nir)^2)`` (Chuvieco et al. 2002);
    - ``MSR`` ``(nir/red - 1)/sqrt(nir/red + 1)`` (Chen 1996);
    - ``CIre`` ``nir/rededge - 1`` (Gitelson et al. 2003);
    - ``NDRE`` ``(nir - rededge)/(nir + rededge)`` (Gitelson and Merzlyak 1994);
    - ``PRI`` ``(r531 - r570)/(r531 + r570)`` (Gamon et al. 1992).

    References
    ----------
    Rouse, J. W., Haas, R. H., Schell, J. A. and Deering, D. W. (1974).
    Monitoring vegetation systems in the Great Plains with ERTS. NASA SP-351.
    Huete, A. et al. (2002). Overview of the radiometric and biophysical
    performance of the MODIS vegetation indices. *Remote Sensing of
    Environment*, 83(1-2), 195-213.
    Chen, J. M. (1996). Evaluation of vegetation indices and a modified
    simple ratio for boreal applications. *Canadian Journal of Remote
    Sensing*, 22(3), 229-242.
    Key, C. H. and Benson, N. C. (2006). Landscape assessment: ground measure
    of severity, the composite burn index; and remote sensing of severity,
    the normalized burn ratio. USDA Forest Service RMRS-GTR-164-CD.

    Examples
    --------
    >>> round(spectral_index("NDVI", nir=0.5, red=0.1), 6)
    0.666667
    >>> [round(v, 6) for v in spectral_index("SAVI", nir=[0.5, 0.4], red=[0.1, 0.1])]
    [0.545455, 0.45]
    """
    if index not in _INDICES:
        raise ValueError(f"index must be one of {_INDICES}")
    vec = [k for k, v in bands.items() if isinstance(v, (list, tuple))]
    if not vec:
        return _one(index, {k: float(v) for k, v in bands.items()})
    n = len(bands[vec[0]])
    if any(len(bands[k]) != n for k in vec):
        raise ValueError("band sequences must have equal length")
    return [_one(index, {k: float(v[i]) if k in vec else float(v) for k, v in bands.items()}) for i in range(n)]


def red_edge_position(r670: float, r700: float, r740: float, r780: float) -> float:
    r"""Red-edge inflection point by linear interpolation (Guyot and Baret 1988).

    ``REP = 700 + 40 ((r670 + r780)/2 - r700) / (r740 - r700)`` nm.

    References
    ----------
    Guyot, G. and Baret, F. (1988). Utilisation de la haute resolution
    spectrale pour suivre l'etat des couverts vegetaux. *Proceedings of the
    4th International Colloquium on Spectral Signatures of Objects in Remote
    Sensing*, ESA SP-287, 279-286.

    Examples
    --------
    >>> round(red_edge_position(0.05, 0.1, 0.35, 0.45), 6)
    724.0
    """
    return 700.0 + 40.0 * ((r670 + r780) / 2.0 - r700) / (r740 - r700)


def toa_reflectance(dn, mult: float, add: float, sun_elevation: float):
    r"""Landsat 8/9 top-of-atmosphere reflectance ``(M_rho Q + A_rho) / sin(theta_SE)``.

    ``mult`` and ``add`` are the band's ``REFLECTANCE_MULT/ADD`` metadata and
    ``sun_elevation`` is in degrees (USGS Landsat 8 Data Users Handbook,
    section 5.3).

    Examples
    --------
    >>> round(toa_reflectance(10000, 2e-5, -0.1, 30.0), 6)
    0.2
    """
    s = math.sin(math.radians(sun_elevation))
    if isinstance(dn, (list, tuple)):
        return [(mult * float(q) + add) / s for q in dn]
    return (mult * float(dn) + add) / s


def fractional_vegetation_cover(ndvi, ndvi_soil: float = 0.2, ndvi_veg: float = 0.5, *, squared: bool = True):
    r"""Fractional vegetation cover from NDVI, clipped to ``[0, 1]``.

    ``((NDVI - NDVI_s)/(NDVI_v - NDVI_s))^2`` (Carlson and Ripley 1997;
    Sobrino et al. 2004's proportion of vegetation) or, with ``squared =
    False``, the linear mixture ``(NDVI - NDVI_s)/(NDVI_v - NDVI_s)``
    (Gutman and Ignatov 1998).

    References
    ----------
    Carlson, T. N. and Ripley, D. A. (1997). On the relation between NDVI,
    fractional vegetation cover, and leaf area index. *Remote Sensing of
    Environment*, 62(3), 241-252.
    Gutman, G. and Ignatov, A. (1998). The derivation of the green vegetation
    fraction from NOAA/AVHRR data for use in numerical weather prediction
    models. *International Journal of Remote Sensing*, 19(8), 1533-1543.

    Examples
    --------
    >>> round(fractional_vegetation_cover(0.35), 6)
    0.25
    """

    def one(v):
        f = min(1.0, max(0.0, (float(v) - ndvi_soil) / (ndvi_veg - ndvi_soil)))
        return f * f if squared else f

    return [one(v) for v in ndvi] if isinstance(ndvi, (list, tuple)) else one(ndvi)


def ndvi_emissivity(ndvi, red, *, ndvi_soil: float = 0.2, ndvi_veg: float = 0.5, veg_emissivity: float = 0.99):
    r"""Land surface emissivity by the NDVI thresholds method (Sobrino, Jimenez-Munoz and Paolini 2004).

    Bare soil (``NDVI < NDVI_s``): ``0.979 - 0.035 red``; full vegetation
    (``NDVI > NDVI_v``): ``veg_emissivity``; mixed pixels: ``0.004 P_v +
    0.986`` with ``P_v`` the squared vegetation proportion
    (:func:`fractional_vegetation_cover`).

    References
    ----------
    Sobrino, J. A., Jimenez-Munoz, J. C. and Paolini, L. (2004). Land surface
    temperature retrieval from LANDSAT TM 5. *Remote Sensing of Environment*,
    90(4), 434-440.

    Examples
    --------
    >>> round(ndvi_emissivity(0.35, 0.08), 6)
    0.987
    """

    def one(v, r):
        v = float(v)
        if v < ndvi_soil:
            return 0.979 - 0.035 * float(r)
        if v > ndvi_veg:
            return veg_emissivity
        return 0.004 * fractional_vegetation_cover(v, ndvi_soil, ndvi_veg) + 0.986

    if isinstance(ndvi, (list, tuple)):
        return [one(v, r) for v, r in zip(ndvi, red)]
    return one(ndvi, red)


def land_surface_temperature(bt, emissivity, *, wavelength: float = 10.895):
    r"""Single-channel land surface temperature ``BT / (1 + (lambda BT / rho) ln epsilon)`` (K).

    ``BT`` is the at-sensor brightness temperature (K), ``lambda`` the
    effective wavelength (micrometres; Landsat 8 band 10 by default) and
    ``rho = h c / sigma = 14388`` micrometre K (Artis and Carnahan 1982).

    References
    ----------
    Artis, D. A. and Carnahan, W. H. (1982). Survey of emissivity
    variability in thermography of urban areas. *Remote Sensing of
    Environment*, 12(4), 313-329.

    Examples
    --------
    >>> round(land_surface_temperature(300.0, 0.98), 6)
    301.383173
    """

    def one(t, e):
        t = float(t)
        return t / (1.0 + (wavelength * t / 14388.0) * math.log(float(e)))

    if isinstance(bt, (list, tuple)):
        es = emissivity if isinstance(emissivity, (list, tuple)) else [emissivity] * len(bt)
        return [one(t, e) for t, e in zip(bt, es)]
    return one(bt, emissivity)


def shortwave_albedo(b1, b3, b4, b5, b7):
    r"""Broadband shortwave albedo from Landsat TM/ETM+ surface reflectance (Liang 2001).

    ``0.356 b1 + 0.130 b3 + 0.373 b4 + 0.085 b5 + 0.072 b7 - 0.0018``.

    References
    ----------
    Liang, S. (2001). Narrowband to broadband conversions of land surface
    albedo I: algorithms. *Remote Sensing of Environment*, 76(2), 213-238.

    Examples
    --------
    >>> round(shortwave_albedo(0.1, 0.1, 0.3, 0.2, 0.1), 6)
    0.1829
    """
    return 0.356 * b1 + 0.130 * b3 + 0.373 * b4 + 0.085 * b5 + 0.072 * b7 - 0.0018


def change_vector(before, after) -> RichResult:
    r"""Change vector analysis of two multiband observations (Malila 1980).

    For each pixel (a sequence of band values per date) the change
    magnitude ``sqrt(sum_k (a_k - b_k)^2)`` and, for two bands, the direction
    ``atan2(d_2, d_1)`` in degrees ``[0, 360)``.

    References
    ----------
    Malila, W. A. (1980). Change vector analysis: an approach for detecting
    forest changes with Landsat. *LARS Symposia*, paper 385.

    Examples
    --------
    >>> r = change_vector([[0.1, 0.3]], [[0.4, 0.7]])
    >>> round(r.magnitude[0], 6), round(r.direction[0], 6)
    (0.5, 53.130102)
    """
    B = [[float(v) for v in p] for p in before]
    A = [[float(v) for v in p] for p in after]
    if len(A) != len(B) or any(len(a) != len(b) for a, b in zip(A, B)):
        raise ValueError("before and after must have the same shape")
    mag = [math.sqrt(ssum((a - b) ** 2 for a, b in zip(pa, pb))) for pa, pb in zip(A, B)]
    dirn = [
        math.degrees(math.atan2(pa[1] - pb[1], pa[0] - pb[0])) % 360.0 if len(pa) == 2 else float("nan")
        for pa, pb in zip(A, B)
    ]
    return RichResult(payload={"magnitude": mag, "direction": dirn})


def accuracy_assessment(reference, predicted) -> RichResult:
    r"""Thematic map accuracy from a confusion matrix (Congalton 1991).

    Rows of ``confusion`` are map (predicted) classes and columns reference
    classes (sorted union of labels).  ``overall`` accuracy, Cohen's
    ``kappa = (p_o - p_e)/(1 - p_e)``, ``producers`` accuracy (diagonal over
    column totals), ``users`` accuracy (diagonal over row totals).

    References
    ----------
    Congalton, R. G. (1991). A review of assessing the accuracy of
    classifications of remotely sensed data. *Remote Sensing of
    Environment*, 37(1), 35-46.

    Examples
    --------
    >>> r = accuracy_assessment([1, 1, 2, 2, 2, 1], [1, 2, 2, 2, 1, 1])
    >>> round(r.overall, 6), round(r.kappa, 6)
    (0.666667, 0.333333)
    """
    ref = list(reference)
    pred = list(predicted)
    if len(ref) != len(pred) or not ref:
        raise ValueError("reference and predicted must have equal non-zero length")
    labels = sorted(set(ref) | set(pred))
    ix = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    C = [[0] * k for _ in range(k)]
    for r, p in zip(ref, pred):
        C[ix[p]][ix[r]] += 1
    n = len(ref)
    rows = [sum(C[i]) for i in range(k)]
    cols = [sum(C[i][j] for i in range(k)) for j in range(k)]
    po = sum(C[i][i] for i in range(k)) / n
    pe = sum(rows[i] * cols[i] for i in range(k)) / (n * n)
    return RichResult(
        payload={
            "labels": labels,
            "confusion": C,
            "overall": po,
            "kappa": (po - pe) / (1.0 - pe) if pe < 1 else float("nan"),
            "producers": [C[i][i] / cols[i] if cols[i] else float("nan") for i in range(k)],
            "users": [C[i][i] / rows[i] if rows[i] else float("nan") for i in range(k)],
        }
    )


def cheatsheet() -> str:
    return "spectral_index / toa_reflectance / land_surface_temperature / accuracy_assessment -> remote sensing basics."
