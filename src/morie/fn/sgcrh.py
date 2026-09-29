"""Cressie-Hawkins robust semivariogram estimator"""

from __future__ import annotations

import math

from ._containers import SpatialResult


def cressie_hawkins(data, coords, *, lags=15, cutoff=None, method="cressie"):
    r"""Cressie-Hawkins robust semivariogram estimator.

    ``2 gamma(h) = [N(h)^{-1} sum |Z(s_i) - Z(s_j)|^{1/2}]^4 / (0.457 +
    0.494 / N(h) + 0.045 / N(h)^2)`` over the ``N(h)`` pairs in each
    distance bin (Cressie and Hawkins 1980; Cressie 1993, eq. 2.4.12): the
    fourth power of the mean square-root difference, bias-corrected under
    Gaussianity, resists outlying pairs far better than Matheron's
    estimator. ``method="gstat"`` drops the ``0.045 / N^2`` term, as
    ``gstat::variogram(cressie = TRUE)``. Bins as gstat: ``cutoff`` one
    third of the bounding-box diagonal by default, ``lags`` bins of equal
    width, bin ``k`` holding ``k w < d <= (k + 1) w``.

    Parameters
    ----------
    data : array-like, shape (n,)
        Observations.
    coords : array-like, shape (n, 2) or (n,)
        Locations.
    lags : int
        Number of distance bins.
    cutoff : float, optional
        Largest distance considered.
    method : str
        ``"cressie"`` (with the ``N^{-2}`` term) or ``"gstat"``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the semivariance of the first non-empty bin;
        ``extra`` has ``gamma``, ``np``, ``dist`` (non-empty bins),
        ``cutoff`` and ``width``.

    References
    ----------
    Cressie, N. and Hawkins, D. M. (1980). Robust estimation of the variogram: I. *Mathematical
    Geology*, 12(2), 115-125.

    Cressie, N. (1993). *Statistics for Spatial Data*, revised edition. Wiley, eq. 2.4.12.

    Examples
    --------
    >>> xy = [[0, 0], [1, 0.2], [2.1, 0], [0.1, 1], [1.2, 1.1], [2, 0.9]]
    >>> r = cressie_hawkins([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], xy, lags=3, cutoff=2.0)
    >>> r.extra["np"], [round(g, 12) for g in r.extra["gamma"]]
    ([9, 3], [0.693606443982, 0.502076995737])
    """
    if method not in ("cressie", "gstat"):
        raise ValueError("method must be 'cressie' or 'gstat'")
    z = [float(v) for v in (data.tolist() if hasattr(data, "tolist") else data)]
    C = coords.tolist() if hasattr(coords, "tolist") else coords
    P = [
        [float(c[0]), float(c[1])]
        if isinstance(c, (list, tuple)) and len(c) > 1
        else [float(c[0] if isinstance(c, (list, tuple)) else c), 0.0]
        for c in C
    ]
    n = len(z)
    if len(P) != n:
        raise ValueError("data and coords must have the same length")
    if cutoff is None:
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        cutoff = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 3.0
    w = cutoff / lags
    cnt = [0] * lags
    sq = [0.0] * lags
    ds = [0.0] * lags
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1])
            if d > cutoff or d == 0.0:
                continue
            k = min(int(math.ceil(d / w)) - 1, lags - 1)
            cnt[k] += 1
            sq[k] += abs(z[i] - z[j]) ** 0.5
            ds[k] += d
    keep = [k for k in range(lags) if cnt[k] > 0]
    gam = []
    for k in keep:
        N = cnt[k]
        corr = 0.457 + 0.494 / N + (0.045 / N**2 if method == "cressie" else 0.0)
        gam.append((sq[k] / N) ** 4 / corr / 2.0)
    return SpatialResult(
        name="Cressie-Hawkins robust semivariogram",
        statistic=gam[0] if gam else float("nan"),
        extra={
            "gamma": gam,
            "np": [cnt[k] for k in keep],
            "dist": [ds[k] / cnt[k] for k in keep],
            "cutoff": cutoff,
            "width": w,
        },
    )


short = "sgcrh"
alias = "cressie_hawkins"
quote = "I think, therefore I am. -- Rene Descartes"


def cheatsheet() -> str:
    return "cressie_hawkins(data, coords, lags, cutoff, method) -> robust semivariogram (Cressie-Hawkins 1980)"


# compact alias per ledger/NAMING.md
cressiehawkins = cressie_hawkins
