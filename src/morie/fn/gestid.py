# morie.fn -- function file (rootcoder007/morie)
"""G-estimation of a structural nested mean model."""

from . import _array_core as np

from ._richresult import RichResult
from ._did import add_intercept, logit_fit, logit_predict

__all__ = ["g_estimation_snmm", "g_estimation_snm"]


def g_estimation_snmm(y, d, X, grid=None, n_grid=401, span=None):
    r"""Find the effect that makes treatment independent of the blip.

    The structural nested mean model posits
    :math:`E[Y - \psi D \mid D, X] = E[Y(0) \mid X]`, so the
    "de-blipped" outcome :math:`H(\psi) = Y - \psi D` should be
    independent of :math:`D` given :math:`X` at the true
    :math:`\psi_0`. G-estimation solves

    .. math::
       \sum_i \{D_i - \hat e(X_i)\}\,\{Y_i - \psi D_i\} = 0

    for :math:`\psi`, which for this linear blip has a closed form.

    The logic is the reverse of an outcome model and that is its
    appeal. Rather than modelling :math:`E[Y|D,X]` and reading off a
    coefficient, it searches for the treatment effect that would make
    the data look like a randomised experiment. Under correct
    specification of the PROPENSITY alone it is consistent, and it
    extends to time-varying treatment where standard regression fails
    outright because the confounder is also a mediator.

    ``test_curve`` returns the estimating function across a grid of
    candidate :math:`\psi`. Its root is the estimate, and the width of
    the region where the test does not reject is a confidence
    interval obtained by INVERSION -- which stays valid where a Wald
    interval does not, notably when the effect is weakly identified
    and the curve is nearly flat. ``weakly_identified`` reports a curve
    whose slope is too shallow to locate a root sharply.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
    X : array-like, shape (n, p)
    grid : array-like, optional
        Candidate psi values for the test curve.
    n_grid : int
    span : float, optional
        Half-width of the automatic grid.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``ci_inversion``, ``test_curve``,
        ``grid``, ``weakly_identified``, ``propensity_range``.

    References
    ----------
    Robins (1992), *Biometrika* 79:321-334.
    Vansteelandt and Joffe (2014), *Statistical Science* 29:707-731.
    Naimi, Cole and Kennedy (2017) for the g-methods overview.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 2))
    >>> d = (rng.uniform(size=500) < 1 / (1 + np.exp(-X[:, 0]))).astype(float)
    >>> y = 1.5 * d + X[:, 0] + rng.normal(size=500)
    >>> bool(abs(g_estimation_snmm(y, d, X)["estimate"] - 1.5) < 0.3)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    if dv.size != n or Xa.shape[0] != n:
        raise ValueError("y, d and X must agree in their first dimension.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")

    B = add_intercept(Xa)
    beta, sep = logit_fit(B, dv)
    e = logit_predict(B, beta)
    r = dv - e                                   # residualised treatment
    den = float(r @ dv)
    if abs(den) < 1e-12:
        raise ValueError(
            "the residualised treatment is orthogonal to treatment; psi is "
            "not identified."
        )
    psi = float(r @ yv / den)

    resid = yv - psi * dv
    # sandwich variance of the estimating equation
    u = r * (resid - float(np.mean(resid)))
    se = float(np.sqrt(np.sum(u ** 2)) / abs(den))

    if grid is None:
        w = span if span is not None else max(6.0 * se, 1e-6)
        g = np.linspace(psi - w, psi + w, int(n_grid))
    else:
        g = np.asarray(grid, dtype=float).ravel()
    curve = np.array([float(r @ (yv - t * dv)) for t in g])
    # invert the test: |S(psi)| <= 1.96 * sd(S) is the acceptance region
    sd_s = float(np.sqrt(np.sum((r * (yv - psi * dv)) ** 2)))
    acc = np.abs(curve) <= 1.959963984540054 * sd_s
    ci_inv = ((float(g[acc][0]), float(g[acc][-1])) if acc.any() else None)
    slope = float(abs(np.polyfit(g, curve, 1)[0])) if g.size > 2 else np.nan
    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": psi,
            "psi": psi,
            "se": se,
            "ci": (psi - z * se, psi + z * se),
            "ci_inversion": ci_inv,
            "inversion_note": (
                "obtained by inverting the test rather than from a normal "
                "approximation, so it stays valid where the estimating "
                "function is nearly flat and psi is weakly identified"
            ),
            "test_curve": curve,
            "grid": g,
            "curve_slope": slope,
            "weakly_identified": bool(
                np.isfinite(slope) and slope < 1e-6 * max(abs(den), 1.0)
            ),
            "logic_note": (
                "g-estimation searches for the effect that would make "
                "treatment look randomised given X, rather than modelling "
                "the outcome and reading off a coefficient; consistency "
                "needs the PROPENSITY to be right"
            ),
            "propensity": e,
            "propensity_range": (float(e.min()), float(e.max())),
            "separated": bool(sep),
            "n": int(n),
            "method": "G-estimation of a structural nested mean model",
        }
    )


def cheatsheet():
    return (
        "gestid: g-estimation by solving the residualised score, with a "
        "test-inversion interval for weak identification"
    )


#: Catalogue alias for :func:`g_estimation_snmm`.
g_estimation_snm = g_estimation_snmm
