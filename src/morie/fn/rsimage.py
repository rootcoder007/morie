# morie.fn -- function file (rootcoder007/morie)
"""Multispectral image processing on band grids: tasseled cap, principal components and the minimum noise
fraction, pan-sharpening, topographic and radiometric (DOS / COST) correction with haze estimation, cloud and
cloud-shadow masks, spectral-angle, Gaussian maximum-likelihood and k-means classification, ATGP endmember
extraction, FPAR from NDVI and the temperature-vegetation dryness index (RStoolbox conventions)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = [
    "tasseled_cap",
    "band_pca",
    "mnf_transform",
    "pan_sharpen",
    "topographic_correction",
    "estimate_haze",
    "radiometric_correction",
    "cloud_mask",
    "cloud_shadow_mask",
    "spectral_angle_classify",
    "gaussian_ml_classify",
    "kmeans_classify",
    "atgp_endmembers",
    "fpar_from_ndvi",
    "tvdi",
]

# Tasseled cap coefficients (rows = input bands), as RStoolbox .TCcoefs: Landsat TM (Crist 1985), ETM+ (Huang et
# al. 2002), OLI (Baig et al. 2014), MODIS (Lobser and Cohen 2007), QuickBird (Yarbrough et al. 2012),
# SPOT-5 (Ivits et al. 2008), RapidEye (Schoenert et al. 2014).
_TC = {
    "landsat5tm": [
        [0.2043, -0.1603, 0.0315],
        [0.4158, -0.2819, 0.2021],
        [0.5524, -0.4934, 0.3102],
        [0.5741, 0.7940, 0.1594],
        [0.3124, 0.0002, 0.6806],
        [0.2303, -0.1446, -0.6109],
    ],
    "landsat7etm": [
        [0.3561, -0.3344, 0.2626],
        [0.3972, -0.3544, 0.2141],
        [0.3904, -0.4556, 0.0926],
        [0.6966, 0.6966, 0.0656],
        [0.2286, -0.0242, -0.7629],
        [0.1596, -0.2630, -0.5388],
    ],
    "landsat8oli": [
        [0.3029, -0.2941, 0.1511],
        [0.2786, -0.2430, 0.1973],
        [0.4733, -0.5424, 0.3283],
        [0.5599, 0.7276, 0.3407],
        [0.5080, 0.0713, -0.7117],
        [0.1872, -0.1608, -0.4559],
    ],
    "modis": [
        [0.4395, -0.4064, 0.1147],
        [0.5945, 0.5129, 0.2489],
        [0.2460, -0.2744, 0.2408],
        [0.3918, -0.2893, 0.3132],
        [0.3506, 0.4882, -0.3122],
        [0.2136, -0.0036, -0.6416],
        [0.2678, -0.4169, -0.5087],
    ],
    "quickbird": [[0.319, 0.542, 0.490], [-0.121, -0.331, -0.517], [0.652, 0.375, -0.639], [0.677, -0.675, 0.292]],
    "spot5": [[0.492, -0.196, 0.397], [0.610, -0.389, 0.260], [0.416, 0.896, 0.118], [0.462, -0.084, -0.872]],
    "rapideye": [
        [0.2435, -0.2216, -0.7564],
        [0.3448, -0.2319, -0.3916],
        [0.4881, -0.4622, 0.5049],
        [0.4930, -0.2154, 0.1400],
        [0.5835, 0.7981, 0.0064],
    ],
}
_TC["landsat4tm"] = _TC["landsat5tm"]


def _grids(bands):
    G = [[[float(v) for v in row] for row in (b.tolist() if hasattr(b, "tolist") else b)] for b in bands]
    nr, nc = len(G[0]), len(G[0][0])
    if any(len(g) != nr or any(len(r) != nc for r in g) for g in G):
        raise ValueError("bands must be equal-shape 2-D grids")
    return G, nr, nc


def _pixels(G):
    """Pixel x band rows, row-major."""
    nr, nc = len(G[0]), len(G[0][0])
    return [[g[i][j] for g in G] for i in range(nr) for j in range(nc)]


def _to_grids(P, nr, nc):
    k = len(P[0])
    return [[[P[i * nc + j][b] for j in range(nc)] for i in range(nr)] for b in range(k)]


def _cov(P, center):
    n, k = len(P), len(P[0])
    return [[ssum((r[a] - center[a]) * (r[b] - center[b]) for r in P) / (n - 1) for b in range(k)] for a in range(k)]


def _eigsym(A):
    """Eigen-decomposition of a symmetric matrix, eigenvalues descending, each vector's largest-|entry| positive."""
    w, V = np.linalg.eigh(np.asarray(A, dtype=float))
    w = [float(v) for v in w.tolist()]
    V = [[float(v) for v in r] for r in V.tolist()]
    order = sorted(range(len(w)), key=lambda i: -w[i])
    vals = [w[i] for i in order]
    vecs = []
    for i in order:
        v = [V[r][i] for r in range(len(V))]
        m = max(range(len(v)), key=lambda t: abs(v[t]))
        vecs.append([-x for x in v] if v[m] < 0 else v)
    return vals, vecs  # vecs[i] = i-th eigenvector


def tasseled_cap(bands, sensor: str = "landsat8oli") -> RichResult:
    r"""Tasseled cap brightness, greenness and wetness: each pixel's band vector times the sensor's coefficients.

    ``bands`` are the sensor's reflective bands in order (6 for Landsat
    TM/ETM+/OLI, 7 MODIS, 4 QuickBird / SPOT-5, 5 RapidEye); coefficients as
    ``RStoolbox::tasseledCap`` (Kauth and Thomas 1976; Crist 1985; Huang et al.
    2002; Baig et al. 2014).

    References
    ----------
    Baig, M. H. A., Zhang, L., Shuai, T. and Tong, Q. (2014). Derivation of a
    tasselled cap transformation based on Landsat 8 at-satellite
    reflectance. *Remote Sensing Letters*, 5(5), 423-431.

    Examples
    --------
    >>> r = tasseled_cap([[[0.1]], [[0.1]], [[0.1]], [[0.3]], [[0.2]], [[0.1]]], "landsat8oli")
    >>> round(r.brightness[0][0], 5), round(r.greenness[0][0], 5)
    (0.39377, 0.10851)
    """
    s = sensor.lower()
    if s not in _TC:
        raise ValueError("sensor must be one of " + ", ".join(sorted(_TC)))
    C = _TC[s]
    G, nr, nc = _grids(bands)
    if len(G) != len(C):
        raise ValueError(f"{s} needs {len(C)} bands")
    out = {
        name: [[ssum(G[b][i][j] * C[b][k] for b in range(len(C))) for j in range(nc)] for i in range(nr)]
        for k, name in enumerate(("brightness", "greenness", "wetness"))
    }
    return RichResult(payload=out)


def band_pca(bands, *, spca: bool = False, n_comp: int | None = None) -> RichResult:
    r"""Principal components of a multiband image, as ``RStoolbox::rasterPCA``.

    Mean and covariance over all pixels (divisor ``n - 1``); ``spca`` uses the
    correlation matrix, scaling each band by its population standard
    deviation (``sd sqrt((n-1)/n)``, princomp's convention). Component scores
    are ``(x - mean) / scale`` times the loadings; each loading vector is
    signed so its largest-magnitude entry is positive.

    Examples
    --------
    >>> r = band_pca([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
    >>> [round(v, 6) for v in r.sdev]
    [2.886751, 0.0]
    """
    G, nr, nc = _grids(bands)
    P = _pixels(G)
    n, k = len(P), len(G)
    mu = [ssum(r[b] for r in P) / n for b in range(k)]
    C = _cov(P, mu)
    if spca:
        sd = [math.sqrt(C[b][b]) for b in range(k)]
        C = [[C[a][b] / (sd[a] * sd[b]) for b in range(k)] for a in range(k)]
        scale = [math.sqrt(C_b * (n - 1) / n) for C_b in (s * s for s in sd)]
    else:
        scale = [1.0] * k
    vals, vecs = _eigsym(C)
    m = k if n_comp is None else min(int(n_comp), k)
    S = [[ssum((r[b] - mu[b]) / scale[b] * vecs[c][b] for b in range(k)) for c in range(m)] for r in P]
    return RichResult(
        payload={
            "scores": _to_grids(S, nr, nc),
            "loadings": [vecs[c] for c in range(m)],
            "sdev": [math.sqrt(max(v, 0.0)) for v in vals],
            "center": mu,
            "scale": scale,
        }
    )


def mnf_transform(bands, *, n_comp: int | None = None) -> RichResult:
    r"""Minimum noise fraction (Green et al. 1988): components ordered by signal-to-noise ratio.

    Noise is estimated from horizontal neighbour differences, ``N = cov(x_ij
    - x_i,j+1) / 2`` (the shift-difference estimator). With ``Sigma`` the data
    covariance, the MNF loadings solve ``Sigma v = lambda N v`` (``lambda =
    1 / noise fraction``, largest first), normalised to unit noise variance
    (``v' N v = 1``) and signed so the largest entry is positive; scores are
    ``(x - mean) v``.

    References
    ----------
    Green, A. A., Berman, M., Switzer, P. and Craig, M. D. (1988). A
    transformation for ordering multispectral data in terms of image
    quality with implications for noise removal. *IEEE Transactions on
    Geoscience and Remote Sensing*, 26(1), 65-74.

    Examples
    --------
    >>> b1 = [[1.0, 2.0, 1.5, 3.0], [2.0, 2.5, 3.5, 4.0], [3.0, 3.2, 4.1, 5.0]]
    >>> b2 = [[0.5, 1.1, 0.9, 1.4], [1.2, 1.1, 1.9, 2.2], [1.4, 1.8, 2.0, 2.9]]
    >>> r = mnf_transform([b1, b2])
    >>> r.snr[0] >= r.snr[1]
    True
    """
    G, nr, nc = _grids(bands)
    k = len(G)
    D = [[G[b][i][j] - G[b][i][j + 1] for b in range(k)] for i in range(nr) for j in range(nc - 1)]
    md = [ssum(r[b] for r in D) / len(D) for b in range(k)]
    N = [[v / 2 for v in row] for row in _cov(D, md)]
    P = _pixels(G)
    mu = [ssum(r[b] for r in P) / len(P) for b in range(k)]
    S = _cov(P, mu)
    # whiten the noise: N = U diag(e) U', W = U diag(e^-1/2)
    ev, U = _eigsym(N)
    W = [[U[c][a] / math.sqrt(ev[c]) for c in range(k)] for a in range(k)]  # a x c
    Sw = [[ssum(W[a][c] * S[a][b] * W[b][d] for a in range(k) for b in range(k)) for d in range(k)] for c in range(k)]
    lam, V = _eigsym(Sw)
    L = [[ssum(W[a][c] * V[q][c] for c in range(k)) for a in range(k)] for q in range(k)]
    L = [[-x for x in v] if v[max(range(k), key=lambda t: abs(v[t]))] < 0 else v for v in L]
    m = k if n_comp is None else min(int(n_comp), k)
    scores = [[ssum((r[a] - mu[a]) * L[q][a] for a in range(k)) for q in range(m)] for r in P]
    return RichResult(payload={"scores": _to_grids(scores, nr, nc), "loadings": L[:m], "snr": lam[:m], "noise_cov": N})


def pan_sharpen(ms, pan, *, method: str = "brovey", rgb=(0, 1, 2)) -> RichResult:
    r"""Pan-sharpen multispectral bands already resampled onto the panchromatic grid.

    ``brovey``: ``ms_k pan / sum(ms_rgb)`` for the ``rgb`` bands (Gillespie et
    al. 1987); ``ihs``: forward IHS transform of the ``rgb`` bands, intensity
    replaced by ``pan`` matched to the intensity's mean and standard
    deviation, inverse transform (Carper et al. 1990); ``pca``: principal
    components of all bands, PC1 replaced by ``pan`` linearly stretched to
    PC1's [min, max], back-transformed (Chavez et al. 1991) -- the Brovey and
    PCA variants as ``RStoolbox::panSharpen``.

    Examples
    --------
    >>> r = pan_sharpen([[[1.0]], [[2.0]], [[1.0]]], [[8.0]])
    >>> [b[0][0] for b in r.bands]
    [2.0, 4.0, 2.0]
    """
    G, nr, nc = _grids(ms)
    Pn = [[float(v) for v in row] for row in (pan.tolist() if hasattr(pan, "tolist") else pan)]
    if method == "brovey":
        out = []
        for b in rgb:
            out.append([[G[b][i][j] * Pn[i][j] / ssum(G[c][i][j] for c in rgb) for j in range(nc)] for i in range(nr)])
        return RichResult(payload={"bands": out})
    if method == "ihs":
        r_, g_, b_ = (G[c] for c in rgb)
        s6, s2 = math.sqrt(6), math.sqrt(2)
        I = [[(r_[i][j] + g_[i][j] + b_[i][j]) / 3 for j in range(nc)] for i in range(nr)]  # noqa: E741
        v1 = [[(r_[i][j] + g_[i][j] - 2 * b_[i][j]) / s6 for j in range(nc)] for i in range(nr)]
        v2 = [[(r_[i][j] - g_[i][j]) / s2 for j in range(nc)] for i in range(nr)]
        fi = [v for row in I for v in row]
        fp = [v for row in Pn for v in row]
        mi, mp = ssum(fi) / len(fi), ssum(fp) / len(fp)
        si = math.sqrt(ssum((v - mi) ** 2 for v in fi) / (len(fi) - 1))
        sp = math.sqrt(ssum((v - mp) ** 2 for v in fp) / (len(fp) - 1))
        Im = [[(Pn[i][j] - mp) / sp * si + mi for j in range(nc)] for i in range(nr)]
        R = [[Im[i][j] + v1[i][j] / s6 + v2[i][j] / s2 for j in range(nc)] for i in range(nr)]
        Gr = [[Im[i][j] + v1[i][j] / s6 - v2[i][j] / s2 for j in range(nc)] for i in range(nr)]
        Bl = [[Im[i][j] - 2 * v1[i][j] / s6 for j in range(nc)] for i in range(nr)]
        return RichResult(payload={"bands": [R, Gr, Bl]})
    if method == "pca":
        pc = band_pca(G)
        S = [[v for row in grid for v in row] for grid in pc.scores]
        s1 = S[0]
        fp = [v for row in Pn for v in row]
        lo, hi, plo, phi = min(s1), max(s1), min(fp), max(fp)
        S[0] = [(v - plo) / (phi - plo) * (hi - lo) + lo for v in fp]
        k = len(G)
        out = [
            [
                [ssum(S[c][i * nc + j] * pc.loadings[c][b] for c in range(k)) + pc.center[b] for j in range(nc)]
                for i in range(nr)
            ]
            for b in range(k)
        ]
        return RichResult(payload={"bands": out})
    raise ValueError("method must be brovey, ihs or pca")


def _ols(x, y):
    mx, my = ssum(x) / len(x), ssum(y) / len(y)
    b = ssum((a - mx) * (c - my) for a, c in zip(x, y)) / ssum((a - mx) ** 2 for a in x)
    return my - b * mx, b


def topographic_correction(
    bands, slope, aspect, *, sun_azimuth: float, sun_zenith: float, method: str = "C"
) -> RichResult:
    r"""Topographic (illumination) correction of reflectance bands, as ``RStoolbox::topCor``.

    Illumination ``cos i = cos z cos s + sin z sin s cos(a_sun - a)`` (angles in
    radians). ``cos``: ``L cos z / cos i`` (Teillet et al. 1982); ``avgcos``:
    ``L + L (mean(cos i) - cos i) / mean(cos i)``; ``minnaert``: ``L (cos z /
    cos i)^k``... applied as ``L cos z / cos i^k`` with ``k`` the slope of ``ln L``
    on ``ln(cos i / cos z)`` over pixels with slope > 2 degrees and ``cos i >=
    0`` (clamped to [0, 1]); ``C``: ``L (cos z + c) / (cos i + c)``, ``c =
    b / m`` from the regression ``L = b + m cos i``; ``stat``: ``L - m cos i``.

    References
    ----------
    Teillet, P. M., Guindon, B. and Goodenough, D. G. (1982). On the
    slope-aspect correction of multispectral scanner data. *Canadian Journal
    of Remote Sensing*, 8(2), 84-106.

    Examples
    --------
    >>> r = topographic_correction([[[0.2, 0.3]]], [[0.0, 0.0]], [[0.0, 0.0]], sun_azimuth=2.0, sun_zenith=0.5,
    ...                            method="cos")
    >>> [round(v, 12) for v in r.bands[0][0]]
    [0.2, 0.3]
    """
    G, nr, nc = _grids(bands)
    S = [[float(v) for v in row] for row in (slope.tolist() if hasattr(slope, "tolist") else slope)]
    A = [[float(v) for v in row] for row in (aspect.tolist() if hasattr(aspect, "tolist") else aspect)]
    cz, sz = math.cos(sun_zenith), math.sin(sun_zenith)
    il = [
        [cz * math.cos(S[i][j]) + sz * math.sin(S[i][j]) * math.cos(sun_azimuth - A[i][j]) for j in range(nc)]
        for i in range(nr)
    ]
    fl = [v for row in il for v in row]
    fs = [v for row in S for v in row]
    out, coef = [], []
    for g in G:
        fx = [v for row in g for v in row]
        if method == "cos":
            res = [x * cz / c for x, c in zip(fx, fl)]
        elif method == "avgcos":
            m = ssum(fl) / len(fl)
            res = [x + x * (m - c) / m for x, c in zip(fx, fl)]
        elif method == "minnaert":
            sel = [t for t in range(len(fx)) if fs[t] > 2 * math.pi / 180 and fl[t] >= 0]
            yv = [math.log(fx[t] if fx[t] > 0 else 1e-32) for t in sel]
            xv = [math.log(fl[t] / cz) for t in sel]
            k = min(1.0, max(0.0, _ols(xv, yv)[1]))
            coef.append(k)
            res = [x * cz / c**k for x, c in zip(fx, fl)]
        elif method in ("C", "stat"):
            b0, b1 = _ols(fl, fx)
            coef.append((b0, b1))
            if method == "C":
                ck = b0 / b1
                res = [x * (cz + ck) / (c + ck) for x, c in zip(fx, fl)]
            else:
                res = [x - b1 * c for x, c in zip(fx, fl)]
        else:
            raise ValueError("method must be cos, avgcos, minnaert, C or stat")
        out.append([res[i * nc : (i + 1) * nc] for i in range(nr)])
    return RichResult(payload={"bands": out, "illumination": il, "coefficients": coef})


def estimate_haze(dn, *, dark_prop: float = 0.01, max_slope: bool = True) -> float:
    r"""Starting haze value (dark-object DN) of one band, as ``RStoolbox::estimateHaze``.

    Over the frequency table of positive DNs, take the last DN whose
    cumulative share is below ``dark_prop``; with ``max_slope`` the DN at the
    steepest rise (maximum second difference of the frequencies smoothed by a
    centred moving average of odd width ``2 floor(idx/20) + 1``, lag-2
    differences as R's ``diff(x, 2)``) up to it
    (Chavez 1988).

    Examples
    --------
    >>> estimate_haze([[0, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10, 12]], dark_prop=0.2, max_slope=False)
    5.0
    """
    vals = [float(v) for row in (dn.tolist() if hasattr(dn, "tolist") else dn) for v in row]
    tab: dict[float, int] = {}
    for v in vals:
        if v == v and v > 0:
            tab[v] = tab.get(v, 0) + 1
    keys = sorted(tab)
    tot = ssum(tab.values())
    freq = [tab[k] / tot for k in keys]
    cum, idx = 0.0, 0
    for i, f in enumerate(freq):
        cum += f
        if cum < dark_prop:
            idx = i + 1
    idx = max(idx, 1)
    if not (max_slope and idx > 1):
        return keys[idx - 1]
    n = 2 * math.floor((idx / 10) / 2) + 1
    h = n // 2
    sm = [ssum(freq[i - h : i + h + 1]) / n if i - h >= 0 and i + h < idx else math.nan for i in range(idx)]
    d2 = [sm[i + 2] - sm[i] for i in range(len(sm) - 2)]  # R diff(x, 2): lag-2 difference
    finite = [i for i, v in enumerate(d2) if v == v]
    best = max(finite, key=lambda i: (d2[i], -i)) if finite else 0
    return keys[min(best + 1, idx - 1)]


_ATMOS = {"veryClear": 4.0, "clear": 2.0, "moderate": 1.0, "hazy": 0.7, "veryHazy": 0.5}


def radiometric_correction(
    dn,
    gain,
    offset,
    *,
    method: str = "apref",
    esun=None,
    sun_elevation: float = 90.0,
    distance: float = 1.0,
    haze_dn=None,
    haze_band: int = 0,
    wavelengths=None,
    atmosphere: str | None = None,
    radiometric_bits: int = 8,
    clamp: bool = True,
    view_zenith: float = 1.0,
) -> RichResult:
    r"""DN to radiance, top-of-atmosphere or dark-object-subtracted surface reflectance, as ``RStoolbox::radCor``.

    Radiance ``L = gain DN + offset``. Reflectance ``rho = C (L - L_haze)``
    with ``C = pi d^2 / (T_v (E_sun cos theta_z T_z))``: ``apref`` (no haze,
    ``T = 1``); ``sdos`` (per-band haze from ``haze_dn``, a list per band;
    ``L_haze = gain_h DN_h + offset_h - 0.01 E_sun cos theta_z / (pi d^2)``);
    ``dos`` (one starting haze value in ``haze_band`` spread to the other bands
    by Chavez's relative scattering model ``(lambda_h / lambda_b)^p``, ``p`` =
    4, 2, 1, 0.7, 0.5 for very clear to very hazy); ``costz`` (as ``dos`` with
    ``T_z = cos theta_z`` and view transmittance ``T_v = cos(view_zenith)``, 1
    degree as RStoolbox, 0 for strict nadir). Without ``atmosphere`` it is chosen from the haze
    radiance as RStoolbox does. ``clamp`` limits reflectance to [0, 1].

    References
    ----------
    Chavez, P. S. (1988). An improved dark-object subtraction technique for
    atmospheric scattering correction of multispectral data. *Remote Sensing
    of Environment*, 24(3), 459-479.
    Chavez, P. S. (1996). Image-based atmospheric corrections -- revisited
    and improved. *Photogrammetric Engineering and Remote Sensing*, 62(9),
    1025-1036.

    Examples
    --------
    >>> r = radiometric_correction([[[100.0]]], [0.5], [1.0], method="rad")
    >>> r.bands[0][0][0]
    51.0
    """
    G, nr, nc = _grids(dn)
    k = len(G)
    gain = [float(v) for v in gain]
    offset = [float(v) for v in offset]
    if method == "rad":
        out = [
            [[max(0.0, gain[b] * v + offset[b]) if clamp else gain[b] * v + offset[b] for v in row] for row in G[b]]
            for b in range(k)
        ]
        return RichResult(payload={"bands": out})
    if method not in ("apref", "sdos", "dos", "costz"):
        raise ValueError("method must be rad, apref, sdos, dos or costz")
    esun = [float(v) for v in esun]
    ct = math.cos((90 - sun_elevation) * math.pi / 180)
    tz = ct if method == "costz" else 1.0
    tv = math.cos(view_zenith * math.pi / 180) if method == "costz" else 1.0
    lhaze = [0.0] * k
    chosen = None
    if method != "apref":
        if method == "sdos":
            hv = [float(v) if v is not None else 0.0 for v in haze_dn]
            for b in range(k):
                ldo = 0.01 * esun[b] * ct / (math.pi * distance**2)
                lhaze[b] = max(0.0, (hv[b] * gain[b] + offset[b]) - ldo)
        else:
            h = int(haze_band)
            hv = float(haze_dn if not isinstance(haze_dn, (list, tuple)) else haze_dn[0])
            ldo = 0.01 * esun[h] * ct * tz * tv / (math.pi * distance**2)
            lh = hv * gain[h] + offset[h] - ldo
            chosen = atmosphere
            if chosen is None:
                top = 2**radiometric_bits - 1
                bounds = [(0, 55), (56, 75), (76, 95), (96, 115), (116, 255)]
                names = list(_ATMOS)
                hit = [nm for nm, (a, z) in zip(names, bounds) if a / 255 * top < lh <= z / 255 * top]
                if not hit:
                    raise ValueError("haze radiance outside the atmosphere table; pass `atmosphere`")
                chosen = hit[0]
            p = _ATMOS[chosen]
            wl = [float(v) for v in wavelengths]
            for b in range(k):
                lb = lh * (wl[h] / wl[b]) ** p * gain[b] / gain[h] + offset[b]
                lhaze[b] = max(0.0, lb)
    out = []
    for b in range(k):
        C = math.pi * distance**2 / (tv * esun[b] * ct * tz)
        g, o = C * gain[b], C * (offset[b] - lhaze[b])
        out.append([[min(1.0, max(0.0, g * v + o)) if clamp else g * v + o for v in row] for row in G[b]])
    return RichResult(payload={"bands": out, "haze_radiance": lhaze, "atmosphere": chosen})


def cloud_mask(blue, tir, *, threshold: float = 0.2) -> RichResult:
    r"""Cloud mask from the normalised difference thermal cloud index, as ``RStoolbox::cloudMask``.

    The thermal band is linearly rescaled to the blue band's [min, max];
    ``NDTCI = (blue - tir') / (blue + tir')`` and clouds are ``NDTCI >=
    threshold`` (bright and cold).

    Examples
    --------
    >>> r = cloud_mask([[0.1, 0.9]], [[300.0, 250.0]])
    >>> r.mask
    [[False, True]]
    """
    B = [[float(v) for v in row] for row in (blue.tolist() if hasattr(blue, "tolist") else blue)]
    T = [[float(v) for v in row] for row in (tir.tolist() if hasattr(tir, "tolist") else tir)]
    fb = [v for row in B for v in row]
    ft = [v for row in T for v in row]
    bl, bh, tl, th = min(fb), max(fb), min(ft), max(ft)
    Tn = [[(v - tl) / (th - tl) * (bh - bl) + bl for v in row] for row in T]
    nd = [[(b - t) / (b + t) for b, t in zip(rb, rt)] for rb, rt in zip(B, Tn)]
    return RichResult(payload={"mask": [[v >= threshold for v in row] for row in nd], "ndtci": nd})


def cloud_shadow_mask(cloud, shift, *, dark=None, quantile: float | None = None) -> RichResult:
    r"""Cloud-shadow mask: the cloud mask displaced by the sun-geometry shift ``(dx, dy)`` in cells.

    As ``RStoolbox::cloudShadowMask`` with a known (``preciseShift``) shift:
    cell ``(i, j)`` is shadow when ``(i - dy, j - dx)`` is cloud (``dx`` to the
    east / columns, ``dy`` north, i.e. towards lower row indices). With
    ``dark`` (e.g. the band sum) and ``quantile`` only cells darker than that
    quantile are kept, and clouds themselves are never shadow.

    Examples
    --------
    >>> cloud_shadow_mask([[True, False, False]], (1, 0)).mask
    [[False, True, False]]
    """
    Cm = [[bool(v) for v in row] for row in (cloud.tolist() if hasattr(cloud, "tolist") else cloud)]
    nr, nc = len(Cm), len(Cm[0])
    dx, dy = int(round(shift[0])), int(round(shift[1]))
    M = [[0 <= i + dy < nr and 0 <= j - dx < nc and Cm[i + dy][j - dx] for j in range(nc)] for i in range(nr)]
    if dark is not None:
        D = [[float(v) for v in row] for row in (dark.tolist() if hasattr(dark, "tolist") else dark)]
        fv = sorted(v for row in D for v in row)
        q = fv[min(len(fv) - 1, int(quantile * (len(fv) - 1)))] if quantile is not None else math.inf
        M = [[M[i][j] and D[i][j] < q and not Cm[i][j] for j in range(nc)] for i in range(nr)]
    return RichResult(payload={"mask": M})


def spectral_angle_classify(bands, endmembers) -> RichResult:
    r"""Spectral angle mapper: angle ``arccos(x . e / (|x| |e|))`` to each endmember; class = smallest angle.

    As ``RStoolbox::sam`` (Kruse et al. 1993); classes are 0-based rows of
    ``endmembers``.

    Examples
    --------
    >>> r = spectral_angle_classify([[[1.0, 0.1]], [[0.1, 1.0]]], [[1, 0], [0, 1]])
    >>> r.classes
    [[0, 1]]
    """
    G, nr, nc = _grids(bands)
    E = [[float(v) for v in e] for e in endmembers]
    ang = []
    for e in E:
        ne = math.sqrt(ssum(v * v for v in e))
        ang.append(
            [
                [
                    math.acos(
                        max(
                            -1.0,
                            min(
                                1.0,
                                ssum(G[b][i][j] * e[b] for b in range(len(e)))
                                / (ne * math.sqrt(ssum(G[b][i][j] ** 2 for b in range(len(e))))),
                            ),
                        )
                    )
                    for j in range(nc)
                ]
                for i in range(nr)
            ]
        )
    cls = [[min(range(len(E)), key=lambda c: (ang[c][i][j], c)) for j in range(nc)] for i in range(nr)]
    return RichResult(payload={"classes": cls, "angles": ang})


def gaussian_ml_classify(bands, train, labels, *, priors=None) -> RichResult:
    r"""Gaussian maximum-likelihood classification (quadratic discriminant), the classical remote-sensing classifier.

    Each class gets the mean and unbiased covariance of its training pixels
    (rows of ``train``, bands as columns); a pixel's class maximises ``log
    pi_c - 1/2 log|S_c| - 1/2 (x - m_c)' S_c^{-1} (x - m_c)``, priors default to
    the training proportions (as ``MASS::qda``). Returns classes (sorted label
    order) and posterior probabilities.

    References
    ----------
    Richards, J. A. (2013). *Remote Sensing Digital Image Analysis*, 5th ed.
    Springer, chapter 8.

    Examples
    --------
    >>> tr = [[0.0, 0.0], [0.2, 0.1], [0.1, 0.3], [5.0, 5.0], [5.2, 4.9], [4.8, 5.3]]
    >>> r = gaussian_ml_classify([[[0.1, 4.9]], [[0.2, 5.1]]], tr, ["w", "w", "w", "v", "v", "v"])
    >>> r.classes
    [['w', 'v']]
    """
    G, nr, nc = _grids(bands)
    X = [[float(v) for v in r] for r in train]
    lab = list(labels)
    cls = sorted(set(lab))
    k = len(X[0])
    stats = []
    for c in cls:
        R = [x for x, y in zip(X, lab) if y == c]
        m = [ssum(r[b] for r in R) / len(R) for b in range(k)]
        S = _cov(R, m)
        Si = inverse(S)
        det = float(np.linalg.det(np.asarray(S, dtype=float)))
        pr = len(R) / len(X) if priors is None else float(priors[cls.index(c)])
        stats.append((m, Si, math.log(pr) - 0.5 * math.log(det)))
    out_c, post = [], []
    for i in range(nr):
        rc, rp = [], []
        for j in range(nc):
            x = [G[b][i][j] for b in range(k)]
            sc = []
            for m, Si, cst in stats:
                d = [a - b for a, b in zip(x, m)]
                sc.append(cst - 0.5 * ssum(d[a] * Si[a][b] * d[b] for a in range(k) for b in range(k)))
            mx = max(sc)
            e = [math.exp(s - mx) for s in sc]
            tot = ssum(e)
            rc.append(cls[max(range(len(cls)), key=lambda t: (sc[t], -t))])
            rp.append([v / tot for v in e])
        out_c.append(rc)
        post.append(rp)
    return RichResult(payload={"classes": out_c, "posterior": post, "levels": cls})


def kmeans_classify(bands, centers, *, max_iter: int = 100) -> RichResult:
    r"""Unsupervised classification by Lloyd's k-means from given initial ``centers`` (rows), as ``kmeans(algorithm = "Lloyd")``.

    Pixels are assigned to the nearest centre (ties to the lower index), each
    centre is replaced by its members' mean (an empty cluster keeps its
    centre), until no assignment changes or ``max_iter``. Classes are
    0-based.

    Examples
    --------
    >>> r = kmeans_classify([[[0.0, 0.2, 5.0, 5.2]]], [[1.0], [4.0]])
    >>> r.classes, r.centers
    ([[0, 0, 1, 1]], [[0.1], [5.1]])
    """
    G, nr, nc = _grids(bands)
    P = _pixels(G)
    C = [[float(v) for v in c] for c in centers]
    k = len(G)
    assign = [-1] * len(P)
    it = 0
    for it in range(1, max_iter + 1):
        new = [min(range(len(C)), key=lambda c: (ssum((p[b] - C[c][b]) ** 2 for b in range(k)), c)) for p in P]
        if new == assign:
            it -= 1
            break
        assign = new
        for c in range(len(C)):
            mem = [p for p, a in zip(P, assign) if a == c]
            if mem:
                C[c] = [ssum(p[b] for p in mem) / len(mem) for b in range(k)]
    wss = [
        ssum(ssum((p[b] - C[c][b]) ** 2 for b in range(k)) for p, a in zip(P, assign) if a == c) for c in range(len(C))
    ]
    return RichResult(
        payload={
            "classes": [assign[i * nc : (i + 1) * nc] for i in range(nr)],
            "centers": C,
            "withinss": wss,
            "iterations": it,
        }
    )


def atgp_endmembers(bands, n: int) -> RichResult:
    r"""Automatic target generation process (Ren and Chang 2003): ``n`` endmember pixels by orthogonal projections.

    The first endmember is the pixel of largest norm; each next one maximises
    the norm of the pixel's projection onto the orthogonal complement of the
    endmembers found so far (``P = I - U (U'U)^{-1} U'``). Returns their
    0-based pixel indices (row-major), grid positions and spectra.

    References
    ----------
    Ren, H. and Chang, C.-I. (2003). Automatic spectral target recognition in
    hyperspectral imagery. *IEEE Transactions on Aerospace and Electronic
    Systems*, 39(4), 1232-1249.

    Examples
    --------
    >>> r = atgp_endmembers([[[1.0, 0.0, 0.5]], [[0.0, 2.0, 0.5]]], 2)
    >>> r.index
    [1, 0]
    """
    G, nr, nc = _grids(bands)
    P = _pixels(G)
    k = len(G)
    idx = []
    U: list[list[float]] = []
    for _ in range(int(n)):
        if U:
            UtU = [[ssum(u[a] * v[a] for a in range(k)) for v in U] for u in U]
            Ui = inverse(UtU)

            def proj(x, Ui=Ui, U=U):
                coef = [
                    ssum(Ui[s][t] * ssum(U[t][a] * x[a] for a in range(k)) for t in range(len(U)))
                    for s in range(len(U))
                ]
                return [x[a] - ssum(coef[s] * U[s][a] for s in range(len(U))) for a in range(k)]
        else:

            def proj(x):
                return x

        norms = [ssum(v * v for v in proj(p)) for p in P]
        best = max(range(len(P)), key=lambda t: (norms[t], -t))
        idx.append(best)
        U.append(P[best])
    return RichResult(payload={"index": idx, "position": [(t // nc, t % nc) for t in idx], "spectra": U})


def fpar_from_ndvi(
    ndvi,
    *,
    ndvi_min: float = 0.05,
    ndvi_max: float = 0.95,
    fpar_min: float = 0.001,
    fpar_max: float = 0.95,
    method: str = "average",
) -> RichResult:
    r"""FPAR from NDVI by the linear NDVI and simple-ratio relations of Sellers et al. (1996) / Los et al. (2000).

    ``FPAR_NDVI = (NDVI - NDVI_min)(FPAR_max - FPAR_min)/(NDVI_max - NDVI_min) +
    FPAR_min``; ``FPAR_SR`` the same with ``SR = (1 + NDVI)/(1 - NDVI)``;
    ``method`` returns either or their ``average`` (Los et al. 2000), clamped
    to [FPAR_min, FPAR_max]. The NDVI extremes are the 5th and 98th
    percentiles of bare soil and full cover for the biome.

    References
    ----------
    Los, S. O. et al. (2000). A global 9-yr biophysical land surface dataset
    from NOAA AVHRR data. *Journal of Hydrometeorology*, 1(2), 183-199.

    Examples
    --------
    >>> [round(v, 6) for v in fpar_from_ndvi([0.05, 0.95], method="ndvi").fpar]
    [0.001, 0.95]
    """
    x = [float(v) for v in (ndvi.tolist() if hasattr(ndvi, "tolist") else ndvi)]

    def sr(v):
        return (1 + v) / (1 - v)

    def lin(v, lo, hi):
        return (v - lo) * (fpar_max - fpar_min) / (hi - lo) + fpar_min

    f_n = [lin(v, ndvi_min, ndvi_max) for v in x]
    f_s = [lin(sr(v), sr(ndvi_min), sr(ndvi_max)) for v in x]
    pick = {"ndvi": f_n, "sr": f_s, "average": [(a + b) / 2 for a, b in zip(f_n, f_s)]}
    if method not in pick:
        raise ValueError("method must be ndvi, sr or average")
    return RichResult(payload={"fpar": [min(fpar_max, max(fpar_min, v)) for v in pick[method]]})


def tvdi(ndvi, lst, *, n_bins: int = 10, ndvi_range=None) -> RichResult:
    r"""Temperature-vegetation dryness index (Sandholt, Rasmussen and Andersen 2002) -- a soil-moisture proxy.

    NDVI is split into ``n_bins`` equal bins over ``ndvi_range`` (default its
    observed range); the dry edge ``T_max = a + b NDVI`` is the OLS line
    through each bin's maximum LST at its mean NDVI, the wet edge the minimum
    LST. ``TVDI = (T - T_min) / (a + b NDVI - T_min)``: 0 wet, 1 dry.

    References
    ----------
    Sandholt, I., Rasmussen, K. and Andersen, J. (2002). A simple
    interpretation of the surface temperature/vegetation index space for
    assessment of surface moisture status. *Remote Sensing of Environment*,
    79(2-3), 213-224.

    Examples
    --------
    >>> r = tvdi([0.1, 0.1, 0.9, 0.9], [300.0, 320.0, 295.0, 305.0], n_bins=2)
    >>> [round(v, 6) for v in r.tvdi]
    [0.2, 1.0, 0.0, 1.0]
    """
    N = [float(v) for v in (ndvi.tolist() if hasattr(ndvi, "tolist") else ndvi)]
    T = [float(v) for v in (lst.tolist() if hasattr(lst, "tolist") else lst)]
    lo, hi = (min(N), max(N)) if ndvi_range is None else (float(ndvi_range[0]), float(ndvi_range[1]))
    w = (hi - lo) / n_bins
    xs, ys = [], []
    for b in range(n_bins):
        ix = [t for t, v in enumerate(N) if lo + b * w <= v < lo + (b + 1) * w or (b == n_bins - 1 and v == hi)]
        if ix:
            xs.append(ssum(N[t] for t in ix) / len(ix))
            ys.append(max(T[t] for t in ix))
    a, bb = _ols(xs, ys)
    tmin = min(T)
    return RichResult(
        payload={"tvdi": [(t - tmin) / (a + bb * v - tmin) for t, v in zip(T, N)], "dry_edge": (a, bb), "t_min": tmin}
    )


def cheatsheet() -> str:
    return (
        "tasseled_cap / band_pca / mnf_transform / pan_sharpen / topographic_correction / radiometric_correction "
        "/ cloud_mask / spectral_angle_classify / gaussian_ml_classify / kmeans_classify / atgp_endmembers / "
        "fpar_from_ndvi / tvdi -> multispectral image processing."
    )
