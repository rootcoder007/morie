# morie.fn -- function file (rootcoder007/morie)
"""Normal-kernel (NOMINATE) spatial voting probability"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._rrng_core import pnorm


def _vec(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _d2(a, b):
    return math.fsum((p - q) ** 2 for p, q in zip(a, b))


def normal_vote(x, *, ideal_point=None, status_quo=None, beta: float = 1.0, w: float = 1.0):
    r"""NOMINATE normal-kernel vote probability for ``x`` (Yea) over ``status_quo`` (Nay).

    Gaussian utility ``U(z) = beta exp(-w^2 ||v - z||^2 / 2)`` with normal
    errors (DW-NOMINATE): ``P(Yea) = Phi(U(x) - U(s))``. ``beta`` is the
    signal-to-noise ratio and ``w`` the dimension weight.

    References
    ----------
    Poole, K. T. and Rosenthal, H. (1997). *Congress: A Political-Economic
    History of Roll Call Voting*. Oxford University Press.

    Examples
    --------
    >>> r = normal_vote([0.0], ideal_point=[0.0], status_quo=[1.0])
    >>> round(r.value, 6)
    0.653014
    """
    x = _vec(x)
    v = [0.0] * len(x) if ideal_point is None else _vec(ideal_point)
    s = [0.0] * len(x) if status_quo is None else _vec(status_quo)
    ux = beta * math.exp(-(w**2) * _d2(v, x) / 2)
    us = beta * math.exp(-(w**2) * _d2(v, s) / 2)
    return DescriptiveResult(name="svnrm", value=float(pnorm(ux - us)), extra={"utility_yea": ux, "utility_nay": us})


norm = normal_vote


def cheatsheet() -> str:
    return "normal_vote(x, ideal_point, status_quo) -> NOMINATE normal-kernel vote probability"


# compact alias per ledger/NAMING.md
normalvote = normal_vote
