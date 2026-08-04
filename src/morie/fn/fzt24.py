# morie.fn -- function file (rootcoder007/morie)
"""MISE of the bias-reduced KDFE (Theorem 2.4)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["gekdfmise", "fauzi_thm2_4_mise_brdkdfe"]


def gekdfmise(n, h, a, biasint, varint, r1=None, r2=None):
    r"""MISE of the bias-reduced KDFE (Theorem 2.4).

    Theorem 2.4:

    .. math:: \mathrm{MISE}(\tilde F_X) = h^8a^4\!\int\!\Big[
              \frac{b_2^2-2b_4F_X}{2F_X}\Big]^2\!dx
              + \frac1n\!\int\! F_X(1-F_X)dx
              - \frac hn\Big[\frac{2(a^4+1)}{(a^2-1)^2}r_1+r_2\Big]
              + o\!\big(h^8+\tfrac hn\big).

    Compare the plain KDFE's MISE from Sec. 2.1, which leads with
    :math:`h^4\mu_2^2R(f')/4`: the bias term has gone from :math:`h^4` to
    :math:`h^8` while the variance terms are unchanged in order. That is
    Theorem 2.4's claim in one line.

    Note the third term has lost its :math:`f_X(x)` factor relative to
    (2.3): integrating :math:`f_X` over the line gives 1. So the variance
    gain from smoothing is a pure constant, independent of ``F``, which is
    why Sec. 2.1 can assert dominance over the empirical df for EVERY
    :math:`F_X`.

    The three integrals are the caller's to supply -- they depend on the
    unknown :math:`F_X`, and estimating them here would silently turn an
    exact theoretical quantity into a plug-in with its own error.

    Parameters
    ----------
    n : int
        Sample size.
    h : float
        Bandwidth.
    a : float
        Second smoothing parameter.
    biasint : float
        ``int [(b_2^2 - 2 b_4 F_X) / (2 F_X)]^2 dx``.
    varint : float
        ``int F_X(x)(1 - F_X(x)) dx``.
    r1, r2 : float, optional
        Kernel constants; default to Gaussian ``r_1`` and ``r_2(a)``.

    Returns
    -------
    RichResult
        Keys ``mise``, ``biasterm``, ``varterm``, ``smoothgain``, ``h``, ``a``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 2.4.
    """
    from .fzr1 import kdfr1
    from .fzr2 import kdfr2

    n = int(n)
    h = float(h)
    a = float(a)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if a <= 0 or abs(a - 1.0) < 1e-6:
        raise ValueError("a must be positive and not close to 1.")
    if r1 is None:
        r1 = float(kdfr1()["estimate"])
    if r2 is None:
        r2 = float(kdfr2(a=a)["estimate"])
    biasterm = h ** 8 * a ** 4 * float(biasint)
    varterm = float(varint) / n
    bracket = 2.0 * (a ** 4 + 1.0) / (a * a - 1.0) ** 2 * float(r1) + float(r2)
    gain = h / n * bracket
    return RichResult(
        payload={
            "mise": float(biasterm + varterm - gain),
            "biasterm": float(biasterm),
            "varterm": float(varterm),
            "smoothgain": float(gain),
            "h": h,
            "a": a,
            "method": "MISE of the bias-reduced KDFE (Theorem 2.4)",
        }
    )


fauzi_thm2_4_mise_brdkdfe = gekdfmise


def cheatsheet():
    return "fzt24: MISE with an h^8 bias term instead of h^4 and unchanged variance orders (Thm 2.4)"


# CANONICAL TEST
# >>> r = gekdfmise(n=100, h=0.2, a=2.0, biasint=1.0, varint=0.5)
# >>> abs(r['mise'] - (r['biasterm'] + r['varterm'] - r['smoothgain'])) < 1e-18
# True
