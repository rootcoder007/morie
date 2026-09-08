# morie.fn -- function file (rootcoder007/morie)
"""Characteristic-function inversion for a pmf."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["cfinvpmf", "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9"]


def cfinvpmf(t, phi_re, phi_im, x):
    """Characteristic-function inversion for a pmf.

    p_x = (1/2pi) int_{-pi}^{pi} e^{-itx} phi_X(t) dt   (Deshmukh eq. 4.9).

    Fourier inversion of the characteristic function of an
    integer-valued random variable.  ``t`` is the quadrature grid on
    [-pi, pi] and ``phi_re``/``phi_im`` are phi_X evaluated on it; the
    integral is taken by the trapezoid rule on the supplied grid, so the
    number of nodes -- not a tolerance -- fixes the accuracy.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Characteristic-function inversion for a pmf", payload=_c.cfinvpmf(t=t, phi_re=phi_re, phi_im=phi_im, x=x))


shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9 = cfinvpmf


def cheatsheet():
    return "shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a4e9: Characteristic-function inversion for a pmf"
