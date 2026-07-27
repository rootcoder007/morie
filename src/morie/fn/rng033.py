# morie.fn -- function file (rootcoder007/morie)
"""Equivalent causal continuous-time convolution with swapped arguments."""

from ._richresult import RichResult
from .rng032 import rangayyan_ch3_causal_convolution

__all__ = ["rangayyan_ch3_causal_convolution_alt"]


def rangayyan_ch3_causal_convolution_alt(x, h, dt=1.0):
    r"""Commuted form :math:`y(t) = \int_0^t h(\tau) x(t-\tau)\, d\tau`.

    Identical to :func:`rangayyan_ch3_causal_convolution` with the
    arguments swapped; the pair exists in the text to make the
    commutativity of convolution explicit.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_ch3_causal_convolution(h, x, dt=dt)
    payload = dict(out)
    payload["method"] = "Causal convolution integral, commuted: y(t) = int_0^t h(tau) x(t-tau) dtau"
    return RichResult(payload=payload)


def cheatsheet():
    return "rng033: y(t) = int_0^t h(tau) x(t-tau) dtau -- rng032 commuted"
