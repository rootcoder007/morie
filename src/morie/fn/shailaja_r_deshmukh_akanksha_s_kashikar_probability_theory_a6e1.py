# morie.fn -- function file (rootcoder007/morie)
"""Limit-superior event (infinitely often)."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["limsupio", "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_6_equation_1"]


def limsupio(dev, k):
    """Limit-superior event (infinitely often).

    D_k = int_{n>=1} un_{m>=n} {|X_m - X| >= 1/k}   (Deshmukh eq. 6.1).

    The limit-superior event: |X_n - X| >= 1/k infinitely often.  On a
    realised path of finite length the event is decided by the tail, so
    a path is in D_k exactly when its last observation still exceeds the
    threshold.  ``dev[r][m]`` is |X_m - X| on path r.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Limit-superior event (infinitely often)", payload=_c.limsupio(dev=dev, k=k))


shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_6_equation_1 = limsupio


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a6e1: Limit-superior event (infinitely often)"
