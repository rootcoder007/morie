# morie.fn -- function file (rootcoder007/morie)
"""Simulation diagnostics: turning-bands conditioning, ensembles and band-number convergence,
realisation standardisation, directional variograms of sample paths, and categorical realisation
diagnostics (class proportions, boundary probability, connectivity, indicator variograms)."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .krgsys import krige, kriging_covariance
from .zstbs import turning_bands

__all__ = [
    "conditional_turning_bands",
    "tb_ensemble",
    "tb_band_convergence",
    "standardise_realisations",
    "directional_variogram",
    "class_proportions",
    "boundary_probability",
    "connectivity",
    "indicator_variogram",
]

_KMAP = {"exponential": "Exp", "gaussian": "Gau", "matern": "Mat"}


def _model(cov_model, sill, range_, nu):
    m = {"model": _KMAP[cov_model], "psill": sill, "range": range_}
    if cov_model == "matern":
        m["kappa"] = nu
    return m


def _pts(P):
    return [tuple(float(v) for v in p) for p in P]


def conditional_turning_bands(
    z,
    data_coords,
    coords,
    cov_model: str = "exponential",
    *,
    sill: float = 1.0,
    range_: float = 1.0,
    nu: float = 0.5,
    mean: float = 0.0,
    n_bands: int = 64,
    n_waves: int = 50,
    seed: int = 1,
) -> RichResult:
    r"""Conditional turning-bands simulation by simple-kriging correction of an unconditional realisation (Journel 1974).

    With ``Z_s`` a :func:`~morie.fn.zstbs.turning_bands` realisation at the
    data and the targets (same seed) and simple-kriging weights ``lambda``
    of the model covariance, ``Z_c(x) = mean + lambda'(z - mean) + Z_s(x) -
    lambda'Z_s(data)``: it honours the data (no nugget) and has the
    conditional covariance when ``Z_s`` has the model covariance
    (isotropic models).

    References
    ----------
    Journel, A. G. (1974). Geostatistics for conditional simulation of ore
    bodies. *Economic Geology*, 69(5), 673-687.

    Examples
    --------
    >>> r = conditional_turning_bands([2.0], [(0, 0)], [(0, 0), (3, 0)], mean=1.0)
    >>> round(r.field[0], 10)
    2.0
    """
    zv = [float(v) for v in z]
    D, Q = _pts(data_coords), _pts(coords)
    nd = len(D)
    s = turning_bands(D + Q, cov_model, sill=sill, range_=range_, nu=nu, n_bands=n_bands, n_waves=n_waves, seed=seed)
    f = [float(v) for v in s["field"]]
    m = _model(cov_model, sill, range_, nu)
    kz = krige(zv, D, Q, m, beta=mean)
    ks = krige(f[:nd], D, Q, m, beta=0.0)
    field = [a + f[nd + k] - b for k, (a, b) in enumerate(zip(kz["prediction"], ks["prediction"]))]
    return RichResult(payload={"field": field, "unconditional": f[nd:], "kriging": kz["prediction"]})


def _semivariogram(vals, P, bounds):
    n = len(P)
    g, c = [0.0] * (len(bounds) - 1), [0] * (len(bounds) - 1)
    for i in range(n):
        for j in range(i + 1, n):
            h = math.dist(P[i], P[j])
            for k in range(len(bounds) - 1):
                if bounds[k] < h <= bounds[k + 1]:
                    g[k] += 0.5 * (vals[i] - vals[j]) ** 2
                    c[k] += 1
                    break
    return [a / b if b else math.nan for a, b in zip(g, c)], c


def tb_ensemble(
    coords,
    cov_model: str = "exponential",
    *,
    sill: float = 1.0,
    range_: float = 1.0,
    nu: float = 0.5,
    nsim: int = 20,
    n_bands: int = 64,
    n_waves: int = 50,
    seed: int = 1,
    bounds=None,
) -> RichResult:
    r"""Ensemble statistics of turning-bands realisations: pointwise mean and variance, realisation variances and the ensemble-mean variogram against the model.

    Realisation ``r`` uses seed ``seed + r``. The ensemble-mean empirical
    semivariogram over the distance classes ``bounds`` is compared with the
    model ``sill - C(h)`` at the mean class distance (ergodic fluctuations
    of single realisations; Lantuejoul 2002, chapter 9).

    References
    ----------
    Lantuejoul, C. (2002). *Geostatistical Simulation: Models and
    Algorithms*. Springer.

    Examples
    --------
    >>> r = tb_ensemble([(0, 0), (1, 0), (2, 0)], nsim=3)
    >>> len(r.pointwise_mean), len(r.realisation_variance)
    (3, 3)
    """
    P = _pts(coords)
    n = len(P)
    sims = [
        [
            float(v)
            for v in turning_bands(
                P, cov_model, sill=sill, range_=range_, nu=nu, n_bands=n_bands, n_waves=n_waves, seed=seed + r
            )["field"]
        ]
        for r in range(nsim)
    ]
    pm = [ssum(s[i] for s in sims) / nsim for i in range(n)]
    pv = [ssum((s[i] - pm[i]) ** 2 for s in sims) / (nsim - 1) for i in range(n)]
    rv = [ssum((v - ssum(s) / n) ** 2 for v in s) / (n - 1) for s in sims]
    out = {"realisations": sims, "pointwise_mean": pm, "pointwise_variance": pv, "realisation_variance": rv}
    if bounds is not None:
        b = [float(v) for v in bounds]
        gs = [_semivariogram(s, P, b)[0] for s in sims]
        out["variogram"] = [ssum(g[k] for g in gs) / nsim for k in range(len(b) - 1)]
        m = _model(cov_model, sill, range_, nu)
        mids = []
        for k in range(len(b) - 1):
            hs = [
                math.dist(P[i], P[j])
                for i in range(n)
                for j in range(i + 1, n)
                if b[k] < math.dist(P[i], P[j]) <= b[k + 1]
            ]
            mids.append(ssum(hs) / len(hs) if hs else math.nan)
        out["model_variogram"] = [sill - kriging_covariance(h, m) if h == h else math.nan for h in mids]
    return RichResult(payload=out)


def tb_band_convergence(
    coords,
    cov_model: str = "exponential",
    *,
    sill: float = 1.0,
    range_: float = 1.0,
    nu: float = 0.5,
    bands=(4, 16, 64),
    nsim: int = 50,
    n_waves: int = 50,
    seed: int = 1,
) -> RichResult:
    r"""Covariance reproduction of turning bands as a function of the number of bands.

    For each ``L`` in ``bands``, ``nsim`` realisations (seeds ``seed + r``)
    give the ensemble covariance of every pair of ``coords``; the root mean
    square difference from the model covariance measures the banding
    artefact plus Monte Carlo error (it falls with ``L``; Mantoglou and
    Wilson 1982; Emery and Lantuejoul 2006).

    References
    ----------
    Emery, X. and Lantuejoul, C. (2006). TBSIM: a computer program for
    conditional simulation of three-dimensional Gaussian random fields via
    the turning bands method. *Computers and Geosciences*, 32(10),
    1615-1628.

    Examples
    --------
    >>> r = tb_band_convergence([(0, 0), (1, 0)], bands=(2, 8), nsim=4)
    >>> len(r.rmse)
    2
    """
    P = _pts(coords)
    n = len(P)
    m = _model(cov_model, sill, range_, nu)
    rmse = []
    for L in bands:
        sims = [
            [
                float(v)
                for v in turning_bands(
                    P, cov_model, sill=sill, range_=range_, nu=nu, n_bands=int(L), n_waves=n_waves, seed=seed + r
                )["field"]
            ]
            for r in range(nsim)
        ]
        mu = [ssum(s[i] for s in sims) / nsim for i in range(n)]
        err = []
        for i in range(n):
            for j in range(i, n):
                c = ssum((s[i] - mu[i]) * (s[j] - mu[j]) for s in sims) / (nsim - 1)
                err.append(c - kriging_covariance(math.dist(P[i], P[j]), m))
        rmse.append(math.sqrt(ssum(e * e for e in err) / len(err)))
    return RichResult(payload={"bands": list(bands), "rmse": rmse})


def standardise_realisations(realisations, *, mean: float = 0.0, variance: float = 1.0) -> RichResult:
    r"""Affine correction of each realisation to a target mean and variance (removing ergodic and banding variance artefacts).

    ``z' = mean + (z - zbar) sqrt(variance / s^2)`` with the realisation's
    sample mean and variance (``n - 1``); the spatial correlation structure
    is untouched (Journel and Xu 1994 discuss such post-processing and its
    cost in conditioning).

    References
    ----------
    Journel, A. G. and Xu, W. (1994). Posterior identification of histograms
    conditional to local data. *Mathematical Geology*, 26(3), 323-359.

    Examples
    --------
    >>> standardise_realisations([[1.0, 3.0]], mean=0.0, variance=2.0).realisations
    [[-1.0, 1.0]]
    """
    out, scale = [], []
    for s in realisations:
        v = [float(a) for a in s]
        n = len(v)
        m = ssum(v) / n
        sd = math.sqrt(ssum((a - m) ** 2 for a in v) / (n - 1))
        k = math.sqrt(variance) / sd
        out.append([mean + (a - m) * k for a in v])
        scale.append(k)
    return RichResult(payload={"realisations": out, "scale": scale})


def directional_variogram(values, coords, alphas, boundaries, *, tol: float = 22.5) -> RichResult:
    r"""Directional empirical semivariograms, as ``gstat::variogram(alpha = , tol.hor = , boundaries = )``.

    Directions ``alpha`` are degrees clockwise from north (the ``y`` axis);
    a pair belongs to direction ``alpha`` when its axial direction is within
    ``tol`` degrees of it, and to class ``(b_k, b_{k+1}]`` by distance.
    Returns ``gamma``, ``np`` and ``dist`` per direction and class, and the
    anisotropy index ``max / min`` of ``gamma`` over directions per class
    (banding artefacts of turning bands show as spurious anisotropy).

    References
    ----------
    Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat
    package. *Computers and Geosciences*, 30(7), 683-691.

    Examples
    --------
    >>> r = directional_variogram([0.0, 1.0, 0.0, 3.0], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 90], [0, 1.5])
    >>> r.gamma
    [[1.0], [2.5]]
    """
    z = [float(v) for v in values]
    P = _pts(coords)
    b = [float(v) for v in boundaries]
    n = len(P)
    G, N, Dm = [], [], []
    for a in alphas:
        g, c, d = [0.0] * (len(b) - 1), [0] * (len(b) - 1), [0.0] * (len(b) - 1)
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
                h = math.hypot(dx, dy)
                ang = math.degrees(math.atan2(dx, dy)) % 180.0
                diff = abs(ang - float(a) % 180.0)
                diff = min(diff, 180.0 - diff)
                if diff > tol:
                    continue
                for k in range(len(b) - 1):
                    if b[k] < h <= b[k + 1]:
                        g[k] += 0.5 * (z[i] - z[j]) ** 2
                        c[k] += 1
                        d[k] += h
                        break
        G.append([x / y if y else math.nan for x, y in zip(g, c)])
        N.append(c)
        Dm.append([x / y if y else math.nan for x, y in zip(d, c)])
    ai = []
    for k in range(len(b) - 1):
        vals = [G[r][k] for r in range(len(G)) if G[r][k] == G[r][k]]
        ai.append(max(vals) / min(vals) if vals and min(vals) > 0 else math.nan)
    return RichResult(payload={"gamma": G, "np": N, "dist": Dm, "anisotropy_index": ai})


def class_proportions(realisations, classes, target=None) -> RichResult:
    r"""Class proportions of categorical realisations (flattened grids), their ensemble mean and spread, and deviation from ``target``.

    Reproduction of the target (global) proportions is the first check of
    indicator and truncated-Gaussian simulations (Deutsch and Journel 1998,
    section V.1).

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB: Geostatistical Software
    Library and User's Guide*, 2nd edn. Oxford University Press.

    Examples
    --------
    >>> class_proportions([[0, 1, 1, 1], [0, 0, 1, 1]], [0, 1], target=[0.5, 0.5]).mean
    [0.375, 0.625]
    """
    props = [[sum(1 for v in s if v == c) / len(s) for c in classes] for s in realisations]
    R = len(props)
    mean = [ssum(p[k] for p in props) / R for k in range(len(classes))]
    sd = [
        math.sqrt(ssum((p[k] - mean[k]) ** 2 for p in props) / (R - 1)) if R > 1 else 0.0 for k in range(len(classes))
    ]
    out = {"proportions": props, "mean": mean, "sd": sd}
    if target is not None:
        out["deviation"] = [m - float(t) for m, t in zip(mean, target)]
    return RichResult(payload=out)


def _grid(s, nrow, ncol):
    return [list(s[r * ncol : (r + 1) * ncol]) for r in range(nrow)]


def boundary_probability(realisations, nrow: int, ncol: int, classes) -> RichResult:
    r"""Per-cell probability of lying on a class boundary (some four-neighbour of another class) across categorical realisations, with class probabilities and entropy.

    Grids are row-major. High boundary probability and entropy locate
    uncertain contacts between classes (Goovaerts 1997, section 8.5).

    References
    ----------
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
    Oxford University Press.

    Examples
    --------
    >>> r = boundary_probability([[0, 0, 1, 1]], 1, 4, [0, 1])
    >>> r.boundary[0]
    [0.0, 1.0, 1.0, 0.0]
    """
    R = len(realisations)
    B = [[0.0] * ncol for _ in range(nrow)]
    Pc = [[[0.0] * ncol for _ in range(nrow)] for _ in classes]
    for s in realisations:
        g = _grid(s, nrow, ncol)
        for i in range(nrow):
            for j in range(ncol):
                nb = [
                    g[i + a][j + b]
                    for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= i + a < nrow and 0 <= j + b < ncol
                ]
                if any(v != g[i][j] for v in nb):
                    B[i][j] += 1 / R
                for k, c in enumerate(classes):
                    if g[i][j] == c:
                        Pc[k][i][j] += 1 / R
    ent = [
        [
            -ssum(Pc[k][i][j] * math.log(Pc[k][i][j]) for k in range(len(classes)) if Pc[k][i][j] > 0)
            for j in range(ncol)
        ]
        for i in range(nrow)
    ]
    return RichResult(payload={"boundary": B, "class_probability": Pc, "entropy": ent})


def connectivity(realisations, nrow: int, ncol: int, target, *, pairs=(), neighbourhood: int = 4) -> RichResult:
    r"""Connectivity of a class across categorical realisations: connected components, percolation and pairwise connection probabilities.

    Components of cells equal to ``target`` (4- or 8-neighbourhood, found by
    union-find); per realisation the number and sizes of components and
    whether one spans the grid left to right (percolation); across
    realisations the probability that each ``(cell_a, cell_b)`` pair
    (row-major indices) is connected (Deutsch 1998; Renard and Allard 2013).

    References
    ----------
    Renard, P. and Allard, D. (2013). Connectivity metrics for subsurface
    flow and transport. *Advances in Water Resources*, 51, 168-196.

    Examples
    --------
    >>> r = connectivity([[1, 1, 0, 1]], 1, 4, 1, pairs=[(0, 1), (0, 3)])
    >>> r.n_components, r.pair_probability
    ([2], [1.0, 0.0])
    """
    offs = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if neighbourhood == 4
        else tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)
    )
    ncomp, sizes, perc = [], [], []
    conn = [0.0] * len(pairs)
    R = len(realisations)
    for s in realisations:
        N = nrow * ncol
        par = list(range(N))

        def find(a, par=par):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a

        for idx in range(N):
            if s[idx] != target:
                continue
            i, j = divmod(idx, ncol)
            for a, b in offs:
                x, y = i + a, j + b
                if 0 <= x < nrow and 0 <= y < ncol and s[x * ncol + y] == target:
                    ra, rb = find(idx), find(x * ncol + y)
                    if ra != rb:
                        par[max(ra, rb)] = min(ra, rb)
        roots = {}
        for idx in range(N):
            if s[idx] == target:
                r = find(idx)
                roots[r] = roots.get(r, 0) + 1
        ncomp.append(len(roots))
        sizes.append(sorted(roots.values(), reverse=True))
        left = {find(r * ncol) for r in range(nrow) if s[r * ncol] == target}
        right = {find(r * ncol + ncol - 1) for r in range(nrow) if s[r * ncol + ncol - 1] == target}
        perc.append(bool(left & right))
        for k, (a, b) in enumerate(pairs):
            if s[a] == target and s[b] == target and find(a) == find(b):
                conn[k] += 1 / R
    return RichResult(
        payload={
            "n_components": ncomp,
            "sizes": sizes,
            "percolates": perc,
            "percolation_probability": sum(perc) / R,
            "pair_probability": conn,
        }
    )


def indicator_variogram(realisations, nrow: int, ncol: int, target, lags) -> RichResult:
    r"""Experimental indicator semivariograms of ``1(Z = target)`` along the rows (``x``) and columns (``y``) of gridded realisations, and their ensemble mean.

    ``gamma(h) = sum (I(x) - I(x + h))^2 / (2 N(h))`` over grid pairs ``h``
    cells apart, the reproduction check of sequential indicator simulation
    against the input indicator variogram (Deutsch and Journel 1998).

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB: Geostatistical Software
    Library and User's Guide*, 2nd edn. Oxford University Press.

    Examples
    --------
    >>> indicator_variogram([[1, 0, 1, 0]], 1, 4, 1, [1, 2]).x
    [[0.5, 0.0]]
    """
    gx, gy = [], []
    for s in realisations:
        g = [[1.0 if v == target else 0.0 for v in row] for row in _grid(s, nrow, ncol)]
        rx, ry = [], []
        for h in lags:
            h = int(h)
            px = [(g[i][j] - g[i][j + h]) ** 2 for i in range(nrow) for j in range(ncol - h)]
            py = [(g[i][j] - g[i + h][j]) ** 2 for i in range(nrow - h) for j in range(ncol)]
            rx.append(ssum(px) / (2 * len(px)) if px else math.nan)
            ry.append(ssum(py) / (2 * len(py)) if py else math.nan)
        gx.append(rx)
        gy.append(ry)
    R = len(gx)
    return RichResult(
        payload={
            "x": gx,
            "y": gy,
            "mean_x": [ssum(r[k] for r in gx) / R for k in range(len(lags))],
            "mean_y": [ssum(r[k] for r in gy) / R for k in range(len(lags))],
        }
    )


def cheatsheet() -> str:
    return (
        "conditional_turning_bands / tb_ensemble / tb_band_convergence / standardise_realisations / "
        "directional_variogram / class_proportions / boundary_probability / connectivity / indicator_variogram."
    )
