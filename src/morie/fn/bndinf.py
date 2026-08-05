# morie.fn -- function file (rootcoder007/morie)
"""Projection-based confidence interval for a partially identified parameter."""

from . import _bndmi as MI
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_inference"]


def bound_inference(theta, moments, alpha=0.05):
    """Test-inversion confidence set for an interval-identified scalar.

    The confidence set is the sub-level set of the moment-inequality
    criterion, and the reported interval is its projection onto the
    parameter line.  For a scalar with two inequalities at most one can
    bind at any parameter value, so the criterion under the null is the
    square of a single one-sided standard normal and the cutoff is
    ``z_{1-alpha}^2`` in closed form -- no bootstrap, and therefore the
    same number in both language arms.

    Formula: ``CS = {theta : Q_n(theta) <= z_{1-alpha}^2}`` with
    ``Q_n(theta) = [sqrt(n)(mL - theta)/sL]_+^2
                 + [sqrt(n)(theta - mU)/sU]_+^2``, whose endpoints are
    ``mL - z sL / sqrt(n)`` and ``mU + z sU / sqrt(n)``.

    Parameters
    ----------
    theta : array-like
        Candidate parameter values to test.
    moments : array-like, shape (n, 2)
        Interval data, column 0 the lower and column 1 the upper end.
    alpha : float, optional
        Miss probability, default 0.05.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width`` (the closed-form set),
        ``grid_lower``, ``grid_upper`` (its projection onto ``theta``),
        ``n_in_set``, ``cutoff``, ``criterion_min``, ``n``.

    References
    ----------
    Romano, J. P. & Shaikh, A. M. (2008).  Inference for identifiable
    parameters in partially identified econometric models.  Journal of
    Statistical Planning and Inference 138(9), 2786-2807.
    doi:10.1016/j.jspi.2008.03.015.
    The criterion and the level-set definition are equations (4.2) and
    (4.10) of Molinari, F. (2021), Microeconometrics with partial
    identification, Handbook of Econometrics 7A (arXiv:2004.11751
    pp. 89, 97), which is the copy used.
    """
    grid = C.vec(theta)
    if len(grid) == 0:
        raise ValueError("bound_inference: theta grid is empty")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("bound_inference: alpha must lie in (0, 1)")
    yl, yu = MI.interval_data(moments, "bound_inference")
    n, mL, sL, mU, sU = MI.stats(yl, yu)
    z = C.qnorm(1.0 - a)
    cut = z * z
    rn = n ** 0.5
    lo = mL - z * sL / rn
    hi = mU + z * sU / rn
    gl = None
    gh = None
    nin = 0
    qmin = None
    for t in grid:
        q = MI.crit(t, n, mL, sL, mU, sU)
        if qmin is None or q < qmin:
            qmin = q
        if q <= cut:
            nin += 1
            if gl is None or t < gl:
                gl = t
            if gh is None or t > gh:
                gh = t
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "grid_lower": gl if gl is not None else float("nan"),
        "grid_upper": gh if gh is not None else float("nan"),
        "n_in_set": nin, "cutoff": cut, "criterion_min": qmin, "n": n,
        "method": "Inference for partially identified parameters"})


def cheatsheet():
    return "bndinf: test-inversion CI for an interval-identified scalar"
