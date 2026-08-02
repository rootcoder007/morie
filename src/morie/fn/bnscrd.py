# morie.fn -- function file (rootcoder007/morie)
"""Bounds on the RD effect when outcomes are missing near the cutoff."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bound_causal_rd"]


def bound_causal_rd(y, x, cutoff, observed=None, bandwidth=None, y_min=None, y_max=None):
    r"""Sharp RD estimate with worst-case bounds for missing outcomes.

    The local linear RD estimate at the cutoff c is the difference of
    the two boundary intercepts fitted within a bandwidth h,

    .. math:: \hat\tau = \lim_{x \downarrow c} \hat m(x)
              - \lim_{x \uparrow c} \hat m(x).

    When some outcomes are missing, the point estimate is only valid
    under an assumption about them; the Manski-style bounds replace
    every missing outcome by :math:`y_{\min}` and by :math:`y_{\max}`
    in turn (the two extremes that maximise and minimise the
    discontinuity), giving an interval that requires no assumption at
    all about *why* they are missing.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome; entries at missing positions are ignored.
    x : array-like, shape (n,)
        Running variable.
    cutoff : float
        Threshold c.
    observed : array-like of bool, shape (n,), optional
        False marks a missing outcome. Default: all observed (then the
        bounds coincide with the point estimate).
    bandwidth : float, optional
        Local-linear bandwidth. Default: the full support on each side.
    y_min, y_max : float, optional
        Logical outcome range. Default: the observed min and max.

    Returns
    -------
    RichResult
        keys: ``estimate`` (complete cases), ``lower``, ``upper``,
        ``width``, ``n_missing``, ``n_used``, ``bandwidth``,
        ``method``.

    References
    ----------
    Manski, C. F. (1990). Nonparametric bounds on treatment effects.
    *American Economic Review*, 80(2), 319-323.

    Imbens, G. W. & Lemieux, T. (2008). Regression discontinuity
    designs: a guide to practice. *Journal of Econometrics*, 142(2),
    615-635. (local linear estimation at the cutoff)
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = y.size
    if x.size != n:
        raise ValueError("y and x must have equal length.")
    c = float(cutoff)
    obs = np.ones(n, dtype=bool) if observed is None else np.asarray(observed, dtype=bool).ravel()
    if obs.size != n:
        raise ValueError("observed must have one entry per unit.")
    if obs.sum() < 4:
        raise ValueError("need at least 4 observed outcomes.")
    lo_y = float(y[obs].min()) if y_min is None else float(y_min)
    hi_y = float(y[obs].max()) if y_max is None else float(y_max)
    if hi_y < lo_y:
        raise ValueError("y_max must be at least y_min.")

    h = float(bandwidth) if bandwidth is not None else float(np.abs(x - c).max()) + 1e-12
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    inwin = np.abs(x - c) <= h
    right, left = inwin & (x >= c), inwin & (x < c)
    if right.sum() < 2 or left.sum() < 2:
        raise ValueError("need at least 2 points on each side within the bandwidth.")

    def intercept(sel, yy):
        D = np.column_stack([np.ones(sel.sum()), x[sel] - c])
        b, *_ = np.linalg.lstsq(D, yy[sel], rcond=None)
        return float(b[0])

    def tau(yy, sel_mask):
        return intercept(right & sel_mask, yy) - intercept(left & sel_mask, yy)

    est = tau(y, obs)
    y_lo = np.where(obs, y, lo_y)
    y_hi = np.where(obs, y, hi_y)
    # widest discontinuity: fill the right arm high and left arm low, and vice versa
    fill_up = np.where(obs, y, np.where(x >= c, hi_y, lo_y))
    fill_dn = np.where(obs, y, np.where(x >= c, lo_y, hi_y))
    all_in = np.ones(n, dtype=bool)
    upper = tau(fill_up, all_in)
    lower = tau(fill_dn, all_in)

    return RichResult(
        payload={
            "estimate": est,
            "lower": float(lower),
            "upper": float(upper),
            "width": float(upper - lower),
            "n_missing": int((~obs).sum()),
            "n_used": int((inwin & obs).sum()),
            "bandwidth": h,
            "method": "Sharp RD with Manski worst-case bounds for missing outcomes",
        }
    )


def cheatsheet():
    return "bnscrd: local-linear RD at c; missing outcomes filled at y_min/y_max both ways"
