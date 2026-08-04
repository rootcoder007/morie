# morie.fn -- k02 batch (rootcoder007/morie)
"""L'Abbe plot coordinates for a set of two-arm trials.

Source consulted: L'Abbe, K.A., Detsky, A.S. and O'Rourke, K. (1987),
Meta-analysis in clinical research, *Annals of Internal Medicine* 107,
224-233.  Each trial is one point, control-arm risk on the x axis and
treated-arm risk on the y axis; the identity line is "no effect".  The pooled
Mantel-Haenszel risk ratio gives the reference line ``y = RR x`` and the
pooled risk difference the line ``y = x + RD``, both reported here so the
plot can be drawn without refitting.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_labbe_plot"]


def ma_labbe_plot(ai, n1i, ci, n2i):
    """L'Abbe plot coordinates and the pooled reference lines.

    Parameters
    ----------
    ai, n1i : array-like
        Events and sample size in the treated arm.
    ci, n2i : array-like
        Events and sample size in the control arm.

    Returns
    -------
    RichResult
        estimate (Mantel-Haenszel risk ratio), x (control risk),
        y (treated risk), size, risk_difference, log_rr, n, method.
    """
    a = np.atleast_1d(np.asarray(ai, dtype=float))
    n1 = np.atleast_1d(np.asarray(n1i, dtype=float))
    c = np.atleast_1d(np.asarray(ci, dtype=float))
    n2 = np.atleast_1d(np.asarray(n2i, dtype=float))
    ntot = n1 + n2
    p1 = a / n1
    p2 = c / n2
    num = float(np.sum(a * n2 / ntot))
    den = float(np.sum(c * n1 / ntot))
    rr = num / den
    rd = float(np.sum((a * n2 - c * n1) / ntot)) / float(np.sum(n1 * n2 / ntot))
    return RichResult(
        payload={
            "estimate": float(rr),
            "x": p2.tolist(),
            "y": p1.tolist(),
            "size": ntot.tolist(),
            "risk_difference": float(rd),
            "log_rr": float(np.log(rr)),
            "n": int(len(a)),
            "method": "L'Abbe plot coordinates with Mantel-Haenszel reference lines (L'Abbe, Detsky & O'Rourke 1987)",
        }
    )


# CANONICAL TEST
# >>> r = ma_labbe_plot([12, 20], [50, 60], [7, 15], [50, 55])
# >>> assert abs(r["x"][0] - 0.14) < 1e-15
# >>> assert r["estimate"] > 1.0


def cheatsheet():
    return "malab(ai, n1i, ci, n2i): L'Abbe plot coordinates."


malabbeplot = ma_labbe_plot
