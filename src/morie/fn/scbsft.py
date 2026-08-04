# morie.fn -- function file (rootcoder007/morie)
"""Baseline-shift-adjusted treatment effect (Tsiatis ANCOVA)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sc_with_baseline_shift"]


def sc_with_baseline_shift(y, D, X, baseline):
    """Semiparametric-efficient treatment effect adjusting for baseline.

    Randomisation already makes the crude difference in means unbiased,
    so adjusting for baseline is not about removing confounding -- it is
    about removing variance.  Tsiatis augmentation subtracts a fitted
    function of the covariates from each arm separately, which cannot
    introduce bias under randomisation, and the arm-specific fit is what
    keeps the gain when the baseline association differs between arms.

    Formula: with ``m_d(V)`` the arm-``d`` regression of ``y`` on the
    covariates ``V = [X, baseline]``, the estimator is
    ``psi = mean_i [m_1(V_i) - m_0(V_i)]``, and the influence function
    contributes ``D (y - m_1) / pi - (1 - D) (y - m_0) / (1 - pi)``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary randomised treatment.
    X : array-like, shape (n, p)
        Covariates other than the baseline value.
    baseline : array-like, shape (n,)
        Baseline level of the outcome.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``pi`` (observed allocation ratio),
        ``shift`` (arm difference in mean baseline), ``n``.

    References
    ----------
    Tsiatis, A. A., Davidian, M., Zhang, M. & Lu, X. (2008).  Covariate
    adjustment for two-sample treatment comparisons in randomized
    clinical trials: a principled yet flexible approach.  Statistics in
    Medicine 27:4658-4677.  The augmented estimator above is section 3
    of that paper; the general semiparametric theory is Tsiatis (2006),
    Semiparametric Theory and Missing Data, Springer.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    bl = C.vec(baseline)
    n = len(yv)
    Xm = C.mat(X)
    V = [[1.0] + list(Xm[i]) + [bl[i]] for i in range(n)]
    idx1 = [i for i in range(n) if Dv[i] > 0.5]
    idx0 = [i for i in range(n) if Dv[i] <= 0.5]
    b1, _, _, _ = C.lstsq([V[i] for i in idx1], [yv[i] for i in idx1])
    b0, _, _, _ = C.lstsq([V[i] for i in idx0], [yv[i] for i in idx0])
    m1 = [C.dot(V[i], b1) for i in range(n)]
    m0 = [C.dot(V[i], b0) for i in range(n)]
    pi = len(idx1) / n
    psi = sum(m1[i] - m0[i] for i in range(n)) / n
    ic = [Dv[i] * (yv[i] - m1[i]) / pi - (1.0 - Dv[i]) * (yv[i] - m0[i]) / (1.0 - pi)
          + m1[i] - m0[i] - psi for i in range(n)]
    mic = sum(ic) / n
    se = math.sqrt(sum((v - mic) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    shift = (sum(bl[i] for i in idx1) / len(idx1)) - (sum(bl[i] for i in idx0) / len(idx0))
    return RichResult(payload={
        "estimate": psi, "se": se, "pi": pi, "shift": shift, "n": n,
        "method": "Tsiatis covariate-adjusted effect with baseline shift"})


def cheatsheet():
    return "scbsft: Baseline-shift-adjusted treatment effect (Tsiatis ANCOVA)."
