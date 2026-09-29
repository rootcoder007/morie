# morie.fn -- function file (rootcoder007/morie)
"""Generalised Pareto distribution fit (Pickands 1975); peaks-over-threshold.

ML-fits the two-parameter GP

    F(x) = 1 - (1 + xi x / sigma)^{-1/xi},   x > 0

to threshold exceedances ``x = y - u`` for chosen threshold ``u``.
"""

from __future__ import annotations

import math

from . import _evt_core as ev
from ._qpcore import inverse
from ._richresult import RichResult

__all__ = ["generalized_pareto", "gpfit"]


def _score(y, s, k):
    """Analytic score of the GPD log-likelihood ``-n log s - (1 + 1/k) sum log(1 + k y / s)``."""
    n = len(y)
    if abs(k) < 1e-10:  # exponential limit
        return [-n / s + sum(y) / (s * s), sum(v * v for v in y) / (2 * s * s) - sum(y) / s]
    z = [1.0 + k * v / s for v in y]
    if min(z) <= 0:
        raise ValueError("parameters outside the support")
    gs = -n / s + (1.0 + 1.0 / k) * sum(k * v / (s * s) / w for v, w in zip(y, z))
    gk = sum(math.log(w) for w in z) / (k * k) - (1.0 + 1.0 / k) * sum(v / s / w for v, w in zip(y, z))
    return [gs, gk]


def _jac(y, s, k):
    """Analytic Hessian of the GPD log-likelihood (``k != 0``)."""
    n = len(y)
    a = 1.0 + 1.0 / k
    z = [1.0 + k * v / s for v in y]
    hss = n / (s * s) + a * sum(-2 * k * v / (s**3 * w) + k * k * v * v / (s**4 * w * w) for v, w in zip(y, z))
    hsk = -sum(k * v / (s * s * w) for v, w in zip(y, z)) / (k * k) + a * sum(
        v / (s * s * w) - k * v * v / (s**3 * w * w) for v, w in zip(y, z)
    )
    hkk = (
        2.0 * sum(v / (s * w) for v, w in zip(y, z)) / (k * k)
        - 2.0 * sum(math.log(w) for w in z) / k**3
        + a * sum(v * v / (s * s * w * w) for v, w in zip(y, z))
    )
    return [[hss, hsk], [hsk, hkk]]


def generalized_pareto(x, threshold: float | None = None):
    r"""Fit a Generalised Pareto distribution to threshold exceedances.

    Peaks over threshold (Pickands 1975; Coles 2001, ch. 4): the excesses
    ``y = x - u`` of the observations above ``u`` (default the type-7 90th
    percentile) are fitted by maximum likelihood, ``l(sigma, xi) = -n log
    sigma - (1 + 1/xi) sum log(1 + xi y / sigma)``. The Nelder-Mead fit of
    :func:`morie.fn._evt_core.gpd_mle` is refined by Newton steps on the
    analytic score (analytic Hessian) to machine precision; standard errors
    are the square roots of the diagonal of the inverse observed information.

    Parameters
    ----------
    x : array-like
        Raw observations.
    threshold : float, optional
        Threshold ``u``.

    Returns
    -------
    RichResult
        ``scale`` (sigma), ``shape`` (xi), ``threshold``, ``n_exceedances``,
        ``se_sigma``, ``se_xi``, ``loglik``, ``estimate`` (= scale), ``se``.

    References
    ----------
    Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics*,
    3(1), 119-131.

    Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer, sec. 4.3.

    Examples
    --------
    >>> x = [0.2, 1.4, 0.7, 2.9, 0.3, 5.1, 1.1, 0.9, 3.8, 0.5, 2.2, 7.4, 1.8, 0.6, 4.3]
    >>> r = generalized_pareto(x, threshold=0.5)
    >>> round(r["scale"], 8), round(r["shape"], 8)
    (2.59492713, -0.18013629)
    """
    xs = [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]
    if len(xs) < 5:
        return RichResult(payload={"estimate": float("nan"), "n": len(xs), "method": "GP (n<5)"})
    if threshold is None:
        s = sorted(xs)
        h = (len(s) - 1) * 0.9
        lo = math.floor(h)
        threshold = s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])
    y = [v - threshold for v in xs if v > threshold]
    n = len(y)
    if n < 5:
        return RichResult(payload={"estimate": float("nan"), "n": n, "method": "GP (too few exceedances)"})
    f = ev.gpd_mle(y)
    s, k = f["sigma"], f["xi"]
    for _ in range(50):
        g = _score(y, s, k)
        J = _jac(y, s, k)
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        ds = (J[1][1] * g[0] - J[0][1] * g[1]) / det
        dk = (J[0][0] * g[1] - J[1][0] * g[0]) / det
        if s - ds <= 0:
            break
        s, k = s - ds, k - dk
        if abs(ds) < 1e-15 * s and abs(dk) < 1e-15:
            break
    J = _jac(y, s, k)
    cov = [[float(v) for v in r] for r in inverse([[-J[0][0], -J[0][1]], [-J[1][0], -J[1][1]]])]
    ll = ev.gpd_loglik(y, s, k)
    se_s, se_k = math.sqrt(max(cov[0][0], 0.0)), math.sqrt(max(cov[1][1], 0.0))
    return RichResult(
        payload={
            "scale": s,
            "shape": k,
            "threshold": float(threshold),
            "n_exceedances": n,
            "se_sigma": se_s,
            "se_xi": se_k,
            "loglik": ll,
            "estimate": s,
            "se": se_s,
            "method": "GP MLE (Pickands 1975; Coles 2001 sec. 4.3)",
        }
    )


def gpfit(data=None, coords=None, x=None, threshold=None):
    """Generalised Pareto fit of ``x`` (or ``data``) above ``threshold``; see :func:`generalized_pareto`.

    A spatial Gaussian-process fit is not provided here (``coords`` with
    two columns raises); use :func:`morie.fn.vgmods.likfit` or the kriging
    modules for that.

    Examples
    --------
    >>> x = [0.2, 1.4, 0.7, 2.9, 0.3, 5.1, 1.1, 0.9, 3.8, 0.5, 2.2, 7.4, 1.8, 0.6, 4.3]
    >>> round(gpfit(x=x, threshold=0.5)["shape"], 8)
    -0.18013629
    """
    if x is not None:
        return generalized_pareto(x, threshold=threshold)
    if data is None:
        raise ValueError("gpfit needs x (or data)")
    if coords is not None:
        C = coords.tolist() if hasattr(coords, "tolist") else coords
        if C and isinstance(C[0], (list, tuple)) and len(C[0]) > 1:
            raise ValueError("spatial Gaussian-process fitting is not provided by gpfit; use vgmods.likfit")
    return generalized_pareto(data, threshold=threshold)


def cheatsheet():
    return "gpfit(x, threshold=auto): Generalised Pareto MLE (POT)."
