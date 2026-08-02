# morie.fn -- function file (rootcoder007/morie)
"""ARE of the sign test against the Wilcoxon signed-rank test."""

from . import _array_core as np

from ._gb_are import efficacy_are
from ._richresult import RichResult

__all__ = ["gibbons_are_sign_wilcoxon"]


def gibbons_are_sign_wilcoxon(f, cdf=None):
    r"""General-density ARE of the sign test relative to the Wilcoxon
    signed-rank test (Gibbons Ch. 13.3):

    .. math:: \mathrm{ARE}(K, T^+) = \frac{f(0)^2}
              {3\big(\int_{-\infty}^{\infty} f^2(x)\,dx\big)^2}.

    Scale-free: replacing f by its rescaled version leaves the ratio
    unchanged, which the tests verify -- an ARE that moved under
    rescaling would be a units bug.

    Parameters
    ----------
    f : callable
        Density symmetric about 0.
    cdf : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``are``, ``f0``, ``int_f2``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 13.3.
    """
    out = efficacy_are(f)
    return RichResult(
        payload={
            "are": out["sign_vs_wilcoxon"], "f0": out["f0"],
            "int_f2": out["int_f2"],
            "method": "ARE(K, T+) = f(0)^2/[3 (int f^2)^2] (Gibbons Ch. 13.3)",
        }
    )


def cheatsheet():
    return "gb_are1: f(0)^2 / [3 (int f^2)^2]; scale-free by construction"
