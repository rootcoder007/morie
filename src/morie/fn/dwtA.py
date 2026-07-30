# morie.fn -- function file (rootcoder007/morie)
"""Wavelet-domain anomaly detection."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["discrete_wavelet_anomaly"]


def discrete_wavelet_anomaly(x, threshold=None, levels=None, max_span=8):
    r"""Detect anomalies as wavelet coefficients that survive a universal threshold.

    A Haar DWT concentrates a smooth signal into few large coefficients while
    an abrupt local change spreads energy into the fine scales. Coefficients
    exceeding :math:`\\sigma\\sqrt{2\\ln n}`, with :math:`\\sigma` the
    MAD-based noise estimate from the finest level, are flagged and mapped
    back to the time points they cover.

    Scale is the diagnostic, not just the detection. ``level_fired`` records
    the **finest** level at which a coefficient exceeded the threshold, which
    is what localises the anomaly: a spike's energy appears at every scale, so
    taking the coarsest firing level would smear a one-point event across half
    the series.

    A Haar step-change is only visible to coefficients that straddle it. A
    shift falling exactly on a dyadic boundary is invisible at every level --
    an alignment artifact of the basis, not of the data. ``per_level_count``
    exposes this: a genuine change with no coefficient firing anywhere means
    the boundary, not the absence of a change.

    The Haar basis has compact support of length 2, so it localises abrupt
    changes precisely but responds poorly to smooth anomalies -- a slow drift
    produces no large coefficient at any scale and is invisible here.

    Parameters
    ----------
    x : array-like
        Series.
    threshold : float, optional
        Coefficient threshold; defaults to the universal threshold.
    levels : int, optional
        Decomposition depth.
    max_span : int
        Largest coefficient span allowed to flag points. Coarser scales are
        still reported in ``per_level_count`` but do not localise.

    Returns
    -------
    RichResult
        ``anomaly`` (boolean per point), ``score``, ``sigma``,
        ``threshold``, ``level_fired``, ``n_anomalies``.

    References
    ----------
    Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by
        wavelet shrinkage. *Biometrika*, 81(3), 425-455.

    Examples
    --------
    A single spike is caught, and it fires at the finest scale.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(0, 0.2, 256)
    >>> y[128] += 8.0
    >>> r = discrete_wavelet_anomaly(y)
    >>> bool(r["anomaly"][128])
    True

    The spike is localised to a short span, because the finest firing level
    is used rather than the coarsest.

    >>> bool(int(r["n_anomalies"]) < 20)
    True
    >>> int(r["level_fired"][128])
    1

    A step change is detected when it does not sit on a dyadic boundary.

    >>> z = np.r_[rng.normal(0, 0.2, 100), rng.normal(5, 0.2, 156)]
    >>> bool(discrete_wavelet_anomaly(z)["n_anomalies"] > 0)
    True

    The documented artifact: the same shift placed exactly on a dyadic
    boundary is invisible to every Haar coefficient.

    >>> zb = np.r_[rng.normal(0, 0.2, 128), rng.normal(5, 0.2, 128)]
    >>> int(discrete_wavelet_anomaly(zb)["n_anomalies"])
    0

    Clean noise triggers few detections, since the threshold is calibrated to
    it.

    >>> bool(discrete_wavelet_anomaly(rng.normal(0, 1, 256))["n_anomalies"] < 30)
    True
    """
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n0 = x.size
    if n0 < 4:
        raise ValueError("need at least 4 observations")
    n = 1 << int(np.ceil(np.log2(n0)))
    pad = np.r_[x, x[::-1]][:n] if n > n0 else x.copy()
    max_lev = int(np.log2(n))
    levels = max_lev - 1 if levels is None else min(int(levels), max_lev)

    approx = pad
    details = []
    for _ in range(levels):
        even, odd = approx[0::2], approx[1::2]
        approx = (even + odd) / np.sqrt(2.0)
        details.append((even - odd) / np.sqrt(2.0))

    d1 = details[0]
    sigma = float(np.median(np.abs(d1 - np.median(d1))) / 0.6745)
    lam = float(sigma * np.sqrt(2.0 * np.log(n))) if threshold is None else float(threshold)

    # Localisation matters: a spike's energy appears at EVERY scale, so taking
    # the coarsest firing level smears a one-point anomaly across half the
    # series. Detection therefore uses the FINEST level at which a coefficient
    # fires, and records that level.
    score = np.zeros(n)
    fired = np.zeros(n, dtype=int)
    per_level = []
    for lv, d in enumerate(details, start=1):
        span = 2 ** lv
        big = np.abs(d) > lam
        per_level.append(int(big.sum()))
        # Coarse coefficients describe the signal, not an anomaly: a single
        # spike produces a large coefficient at EVERY scale, and the coarsest
        # of them spans half the series. Detection is therefore capped at
        # scales fine enough to localise.
        if span > max_span:
            continue
        for j in np.flatnonzero(big):
            lo, hi = j * span, min((j + 1) * span, n)
            fresh = fired[lo:hi] == 0
            score[lo:hi][fresh] = abs(d[j])
            fired[lo:hi][fresh] = lv
    score, fired = score[:n0], fired[:n0]
    anom = score > lam
    return RichResult(
        title="Wavelet anomaly detection",
        summary_lines=[("n", int(n0)), ("sigma", sigma), ("threshold", lam),
                       ("anomalies", int(anom.sum()))],
        warnings=["the Haar basis localises abrupt changes well but is blind "
                  "to slow drift, which produces no large coefficient at any scale"],
        payload={
            "anomaly": anom, "score": score, "sigma": sigma,
            "threshold": lam, "level_fired": fired, "per_level_count": per_level,
            "n_anomalies": int(anom.sum()), "levels": int(levels),
            "method": "discrete_wavelet_anomaly",
        },
    )


def cheatsheet():
    return "dwtA: wavelet coefficients past sigma*sqrt(2 log n); LEVEL distinguishes spike from level shift"
