# morie.fn -- function file (rootcoder007/morie)
"""LightGCN layer combination."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["lgcnprop", "lightgcn"]


def lgcnprop(A, E, K, alpha=None):
    """LightGCN layer combination.

    LightGCN: e = sum_k alpha_k S^k e, S = D^-1/2 A D^-1/2.

    He et al. (2020).  Feature transformation and nonlinearity are
    dropped entirely -- only neighbourhood averaging remains -- and the
    layer outputs are combined by fixed weights, uniform 1/(K+1) by
    default as in the paper.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="LightGCN layer combination", payload=_c.lgcnprop(A=A, E=E, K=K, alpha=alpha))


lightgcn = lgcnprop


def cheatsheet():
    return "lightG: LightGCN layer combination"
