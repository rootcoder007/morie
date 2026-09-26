"""Expected Matheron semivariogram under a linear drift, eq (5.35)."""

import math

from ._richresult import RichResult
from ._schab_glsvg import lag_pairs, model_gamma

__all__ = ["drift_semivariogram_bias"]


def drift_semivariogram_bias(coords, X, beta, breaks, nugget, sill, range, model="exponential"):
    r"""Expected Matheron estimator when the mean is not constant.

    Schabenberger & Gotway (2005, eq 5.35, p. 255), after Cressie (1993,
    p. 165): under :math:`Z(s) = X(s)\beta + e(s)`,

    .. math::

        E[(Z(s_i) - Z(s_j))^2] = 2\gamma(s_i - s_j)
        + \Big\{\sum_{k=1}^p \beta_k(x_k(s_i) - x_k(s_j))\Big\}^2,

    so the classical estimator averages the true semivariogram and a
    squared drift contrast in each lag class and no longer estimates
    :math:`\gamma`. This returns both parts class by class.

    Parameters
    ----------
    coords : sequence of (x, y)
    X : sequence of rows
        Covariates :math:`x(s_i)'`, one row per site.
    beta : sequence of float
    breaks : sequence of float
        Lag-class boundaries, class m = (breaks[m], breaks[m+1]].
    nugget, sill, range : float
        Semivariogram parameters of the error process.
    model : str

    Returns
    -------
    RichResult
        ``lags`` (mean pair distance), ``npairs``, ``expected``
        (:math:`E[\hat\gamma(h_m)]`), ``semivariogram_part``, ``drift_part``.

    References
    ----------
    Cressie, N. (1993). Statistics for Spatial Data, revised ed. Wiley,
    p. 165. Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods
    for Spatial Data Analysis. Chapman & Hall/CRC, eq (5.35), p. 255.
    """
    b = [float(v) for v in beta]
    xs = [[float(v) for v in row] for row in X]
    pts, classes, lags = lag_pairs(coords, breaks)
    if len(xs) != len(pts) or any(len(r) != len(b) for r in xs):
        raise ValueError("`X` needs one row per site and one column per `beta`")
    mu = [sum(xv * bv for xv, bv in zip(r, b)) for r in xs]
    semi, drift = [], []
    for c in classes:
        d = [math.dist(pts[i], pts[j]) for i, j in c]
        semi.append(sum(model_gamma(d, nugget, sill, range, model)) / len(c))
        drift.append(sum((mu[i] - mu[j]) ** 2 for i, j in c) / (2.0 * len(c)))
    return RichResult(
        title="Expected Matheron semivariogram under drift (eq 5.35)",
        summary_lines=[("classes", len(classes)), ("max drift part", max(drift))],
        payload={
            "lags": lags,
            "npairs": [len(c) for c in classes],
            "expected": [s + t for s, t in zip(semi, drift)],
            "semivariogram_part": semi,
            "drift_part": drift,
        },
    )


def cheatsheet():
    return "vgdrift: E[(Z_i - Z_j)^2] = 2 gamma + (drift contrast)^2, eq 5.35"
