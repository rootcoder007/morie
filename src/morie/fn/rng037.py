# morie.fn -- function file (rootcoder007/morie)
"""Equivalent discrete-time causal convolution with swapped arguments."""

from ._richresult import RichResult
from .rng036 import rangayyan_ch3_discrete_convolution_causal

__all__ = ["rangayyan_ch3_discrete_convolution_causal_alt"]


def rangayyan_ch3_discrete_convolution_causal_alt(x, h, n=None):
    r"""Commuted form :math:`y(n) = \sum_{k=0}^{n} h(k) x(n-k)`.

    Convolution is commutative, so this returns exactly the same
    sequence as :func:`rangayyan_ch3_discrete_convolution_causal` with
    the arguments swapped -- the identity is the point of the pair in
    the text, and the test asserts it.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_ch3_discrete_convolution_causal(h, x, n=n)
    payload = dict(out)
    payload["method"] = "Causal discrete convolution, commuted: y(n) = sum_k h(k) x(n-k)"
    return RichResult(payload=payload)


def cheatsheet():
    return "rng037: y(n) = sum_{k=0}^{n} h(k) x(n-k) -- same as rng036 by commutativity"
