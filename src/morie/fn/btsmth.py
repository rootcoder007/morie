# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Smoothed bootstrap: resample, then perturb by a kernel draw.

Silverman, B. W. and Young, G. A. (1987), "The bootstrap: to smooth or
not to smooth?", *Biometrika* 74(3), 469-479,
doi:10.1093/biomet/74.3.469 (verified against Crossref).

Efron's bootstrap resamples the empirical distribution, which is a step
function; for statistics that depend on the local shape of F (a density
ordinate, a quantile, a mode) that atomicity is the dominant source of
error.  The smoothed bootstrap resamples from a kernel density estimate
instead, which for a Gaussian kernel is exactly

    x*_i = x_{I_i} + h eps_i,     I_i ~ U{1..n},  eps_i ~ N(0, 1),

so no density has to be evaluated -- one resampled point plus one scaled
normal draw per observation.

The bandwidth is the whole trade-off and the paper's title is the
warning: smoothing helps only when the functional is sensitive to the
fine structure of F, and h > 0 inflates the variance of anything that is
not.  Both facts are visible in the closed form for the mean, which is
this module's anchor: the conditional variance of the smoothed bootstrap
mean is exactly

    ( sigma_hat^2 + h^2 ) / n,     sigma_hat^2 = sum (x_i - xbar)^2 / n,

i.e. the ordinary bootstrap variance plus h^2/n, with no benefit.
``var_closed`` reports it and does not run through the resampling loop.
h = 0 reduces the whole procedure to the ordinary bootstrap exactly,
which is the second anchor.

Draws come from the package's shared Lehmer stream, uniforms and normals
interleaved in a fixed order, so both arms perturb identically.
"""

from __future__ import annotations

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["boot_smoothed"]


def boot_smoothed(x, stat=None, h=None, B=200, seed=1, alpha=0.05):
    """Smoothed bootstrap replicates.

    Parameters
    ----------
    x : array-like
        The observed sample.
    stat : callable, optional
        Statistic of a sample.  Defaults to the mean.
    h : float, optional
        Kernel bandwidth.  Defaults to Silverman's rule of thumb,
        ``0.9 min(s, IQR/1.34) n^(-1/5)``.  ``h = 0`` gives the
        ordinary bootstrap.
    B : int
        Replicates.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate.

    Returns
    -------
    RichResult
        ``theta_b``, ``estimate`` (statistic on the data), ``se``,
        ``lo``/``hi``, ``h``, ``var_closed`` (closed-form conditional
        variance of the smoothed mean; NaN for a custom statistic),
        ``n``, ``B``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_smoothed: need at least two observations")
    if int(B) < 2:
        raise ValueError("boot_smoothed: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_smoothed: alpha must lie strictly between 0 and 1")
    if h is None:
        s = core.sd(xx, 1)
        iqr = core.quantile7(xx, 0.75) - core.quantile7(xx, 0.25)
        spread = s if (iqr <= 0.0 or s < iqr / 1.34) else iqr / 1.34
        h = 0.9 * spread * n ** (-0.2)
    h = float(h)
    if h < 0.0:
        raise ValueError("boot_smoothed: h must be non-negative")
    f = core.mean if stat is None else stat
    th = float(f(xx))
    g = C.Lcg(seed)
    theta = []
    for _ in range(int(B)):
        smp = []
        for _i in range(n):
            j = int(g.unif() * n)
            if j >= n:
                j = n - 1
            smp.append(xx[j] + h * g.norm())
        theta.append(float(f(smp)))
    xb = core.mean(xx)
    s2 = sum((u - xb) ** 2 for u in xx) / n
    return RichResult(
        title="Smoothed bootstrap (Silverman and Young 1987)",
        summary_lines=[("n", n), ("h", h), ("B", int(B)), ("estimate", th)],
        payload={
            "theta_b": theta,
            "estimate": th,
            "se": core.sd(theta, 1),
            "lo": core.quantile7(theta, a / 2.0),
            "hi": core.quantile7(theta, 1.0 - a / 2.0),
            "h": h,
            "var_closed": ((s2 + h * h) / n) if stat is None else float("nan"),
            "n": n,
            "B": int(B),
            "method": "Silverman and Young (1987) Biometrika 74(3):469-479",
        },
    )


def cheatsheet():
    return "btsmth: x* = x_I + h*eps; for the mean this only ADDS h^2/n to the variance"
