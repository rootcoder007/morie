# morie.fn -- function file (rootcoder007/morie)
"""Piecewise log-linear shedding curve."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["shedcurve", "viral_shedding_model"]


def shedcurve(days, load, t_peak, t_plateau):
    """Piecewise log-linear shedding curve.

    Piecewise log-linear shedding curve: rise, plateau, decay.

    Viral load is fitted on the log10 scale in three segments split at
    ``t_peak`` and ``t_plateau``: a rising slope before the peak, a flat
    level between, and a decay slope after.  Each segment is an ordinary
    least squares fit, so the result is closed form.  The shape matters
    clinically because peak shedding precedes symptom onset, which is
    the paper's finding.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Piecewise log-linear shedding curve", payload=_c.shedcurve(days=days, load=load, t_peak=t_peak, t_plateau=t_plateau))


viral_shedding_model = shedcurve


def cheatsheet():
    return "shedmd: Piecewise log-linear shedding curve"
