# morie.fn -- function file (rootcoder007/morie)
"""Simplified graph convolution propagation."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["sgcprop", "sgc"]


def sgcprop(A, X, K):
    """Simplified graph convolution propagation.

    Simplified graph convolution: S^K X, S = D^-1/2 (A + I) D^-1/2.

    Wu et al. (2019).  Collapsing the nonlinearities between graph
    convolution layers leaves a fixed linear smoothing operator applied
    K times, which can be precomputed once; the classifier that follows
    is then an ordinary logistic regression.  The result is the
    propagated feature matrix.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Simplified graph convolution propagation", payload=_c.sgcprop(A=A, X=X, K=K))


sgc = sgcprop


def cheatsheet():
    return "sgcL: Simplified graph convolution propagation"
