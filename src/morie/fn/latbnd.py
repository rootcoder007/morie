# morie.fn -- function file (rootcoder007/morie)
"""Bounds on treatment effects without full identification."""

import numpy as np

from ._richresult import RichResult

__all__ = ["late_bounds"]


def late_bounds(y, d, z=None, y_min=None, y_max=None, mono=True):
    r"""Manski-style bounds, and the LATE when an instrument identifies it.

    With a bounded outcome and NO assumptions beyond that, the
    worst-case (Manski) bounds on the ATE come from filling each
    missing potential outcome with its extreme:

    .. math::
       \text{ATE} \in \big[\,
         E[Y|D{=}1]p + y_{\min}(1-p) - y_{\max}p - E[Y|D{=}0](1-p),\;
         \ldots \big]

    Their width is always :math:`y_{\max} - y_{\min}`, and they ALWAYS
    contain zero. That is not a defect of the method -- it is the
    honest content of the data before assumptions. Every narrower
    interval in an observational study is purchased with an assumption,
    and the point of computing these is to see the price.

    With an instrument, the Wald ratio identifies the LATE among
    compliers under exclusion and monotonicity. ``late`` reports it and
    ``complier_share`` the fraction of the sample it speaks for --
    which is often small, and is the quantity omitted when a LATE is
    reported as though it were an ATE.

    ``assumption_cost`` is the ratio of the no-assumption width to the
    LATE interval's width: how many times narrower the answer became
    once the exclusion restriction was imposed.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
    z : array-like of {0, 1}, optional
        Instrument.
    y_min, y_max : float, optional
        Outcome support. Taken from the data when omitted, which
        UNDERSTATES the bounds.
    mono : bool
        Assume monotonicity (no defiers) for the LATE.

    Returns
    -------
    RichResult
        ``bounds``, ``width``, ``contains_zero``, ``late``,
        ``complier_share``, ``assumption_cost``, ``support_from_data``.

    References
    ----------
    Manski (1990), *American Economic Review* 80:319-323.
    Imbens and Angrist (1994), *Econometrica* 62:467-475.
    Manski and Pepper (2000), *Econometrica* 68:997-1010.

    Examples
    --------
    >>> out = late_bounds([0, 1, 0, 1], [0, 0, 1, 1], y_min=0, y_max=1)
    >>> bool(out["contains_zero"])
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    n = yv.size
    if dv.size != n:
        raise ValueError("y and d must agree in length.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if min(int(dv.sum()), int((1 - dv).sum())) < 1:
        raise ValueError("need at least one unit in each treatment arm.")

    from_data = y_min is None or y_max is None
    lo_y = float(np.min(yv)) if y_min is None else float(y_min)
    hi_y = float(np.max(yv)) if y_max is None else float(y_max)
    if hi_y <= lo_y:
        raise ValueError("y_max must exceed y_min.")

    p = float(dv.mean())
    m1 = float(yv[dv == 1].mean())
    m0 = float(yv[dv == 0].mean())
    # E[Y(1)] in [m1*p + lo*(1-p), m1*p + hi*(1-p)], and symmetrically
    e1_lo, e1_hi = m1 * p + lo_y * (1 - p), m1 * p + hi_y * (1 - p)
    e0_lo, e0_hi = m0 * (1 - p) + lo_y * p, m0 * (1 - p) + hi_y * p
    lo, hi = e1_lo - e0_hi, e1_hi - e0_lo
    width = hi - lo

    late = share = cost = None
    late_ci = None
    if z is not None:
        zv = np.asarray(z, dtype=float).ravel()
        if zv.size != n:
            raise ValueError("z has %d entries for %d rows." % (zv.size, n))
        if not np.all(np.isin(zv, (0.0, 1.0))):
            raise ValueError("z must be binary 0/1.")
        m1z, m0z = zv == 1, zv == 0
        if m1z.sum() < 2 or m0z.sum() < 2:
            raise ValueError("need at least 2 units at each instrument value.")
        fs = float(dv[m1z].mean() - dv[m0z].mean())
        rf = float(yv[m1z].mean() - yv[m0z].mean())
        if abs(fs) > 1e-12:
            late = rf / fs
            share = abs(fs)
            n1, n0 = int(m1z.sum()), int(m0z.sum())
            v_rf = yv[m1z].var(ddof=1) / n1 + yv[m0z].var(ddof=1) / n0
            v_fs = dv[m1z].var(ddof=1) / n1 + dv[m0z].var(ddof=1) / n0
            c = (np.cov(yv[m1z], dv[m1z], ddof=1)[0, 1] / n1
                 + np.cov(yv[m0z], dv[m0z], ddof=1)[0, 1] / n0)
            se = float(np.sqrt(max(v_rf / fs ** 2 + rf ** 2 * v_fs / fs ** 4
                                   - 2 * (rf / fs ** 3) * c, 0.0)))
            late_ci = (late - 1.959963984540054 * se,
                       late + 1.959963984540054 * se)
            w = late_ci[1] - late_ci[0]
            cost = float(width / w) if w > 0 else np.inf
    return RichResult(
        payload={
            "estimate": (lo, hi),
            "bounds": (lo, hi),
            "width": float(width),
            "contains_zero": bool(lo <= 0 <= hi),
            "no_assumption_note": (
                "the worst-case width is always y_max - y_min and the bounds "
                "always contain zero; that is the honest content of the data "
                "before assumptions, not a failure of the method"
            ),
            "ey1_bounds": (e1_lo, e1_hi),
            "ey0_bounds": (e0_lo, e0_hi),
            "late": late,
            "late_ci": late_ci,
            "complier_share": share,
            "complier_note": (
                None if share is None else
                "the LATE speaks for the %.1f %% of the sample who comply; "
                "reporting it as an ATE silently generalises beyond them"
                % (100 * share)
            ),
            "assumption_cost": cost,
            "cost_note": (
                None if cost is None else
                "the exclusion restriction and monotonicity together bought "
                "an interval %.1f times narrower than the data alone support"
                % cost
            ),
            "monotonicity": bool(mono),
            "support": (lo_y, hi_y),
            "support_from_data": bool(from_data),
            "support_note": (
                None if not from_data else
                "the outcome support was taken from the observed range, "
                "which UNDERSTATES the true bounds"
            ),
            "n": int(n),
            "method": "Manski worst-case bounds, with the LATE when identified",
        }
    )


def cheatsheet():
    return (
        "latbnd: no-assumption bounds against the instrumented LATE, showing "
        "what the identifying assumptions bought"
    )
