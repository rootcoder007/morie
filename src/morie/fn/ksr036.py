# morie.fn -- function file (rootcoder007/morie)
"""Bracketing Donsker theorem."""

import numpy as np

from ._richresult import RichResult
from .ksr035 import kosorok_ch2_donsker_bracketing_integral

__all__ = ["kosorok_ch2_donsker_bracketing_theorem"]


def kosorok_ch2_donsker_bracketing_theorem(N_bracket, F=None, P=None):
    r"""Bracketing Donsker theorem: if
    :math:`J_{[\,]}(\infty, \mathcal F, L_2(P)) < \infty` then
    :math:`\mathcal F` is P-Donsker.

    Evaluates the entropy integral at a large upper limit and reports
    whether the sufficient condition holds. Sufficient, not necessary:
    a class failing this test may still be Donsker by another route
    (uniform entropy, for instance), so the returned key is named
    ``sufficient_condition_met`` rather than ``is_donsker``.

    Parameters
    ----------
    N_bracket : callable
        eps -> bracketing number in L_2(P).
    F, P : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``J_infinity``, ``sufficient_condition_met``,
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (the bracketing Donsker theorem).
    """
    out = kosorok_ch2_donsker_bracketing_integral(N_bracket, delta=1.0)
    J = out["J"]
    return RichResult(
        payload={"J_infinity": J,
                 "sufficient_condition_met": bool(np.isfinite(J) and J < 1e6),
                 "method": "J_[](inf, F, L2(P)) < inf => P-Donsker (SUFFICIENT only)"}
    )


def cheatsheet():
    return "ksr036: sufficient condition, not a characterisation"
