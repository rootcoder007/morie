# morie.fn -- function file (rootcoder007/morie)
"""Gradient-descent weight change.

Implements eq. (10.10)-(10.11) p.410 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_10", "mvsml_ann_gradient"]


def mvsml_reproducing_kernel_eq_10_10(X, y, W, activations=None, eta=0.1):
    """Delta w_jk^(l) = -eta dE/dw_jk^(l) (eq. 10.10), with the chain
    rule pieces dE/dy-hat = -(y - y-hat) and dy-hat/dz = g^(l)'(z)
    from (10.11).  Moving the weights down the slope of the loss is
    the whole intuition behind backpropagation. Keys: estimate."""
    g = _gp.ann_backprop_gradients(X, y, W, activations)
    upd = [[[-eta * v for v in row] for row in G]
           for G in g["gradients"]]
    res = RichResult(payload={"estimate": g["loss"],
                              "gradients": g["gradients"],
                              "weight_changes": upd,
                              "loss": g["loss"],
                              "method": "gradient-descent weight change (MVSML 2022 eq. 10.10-10.11)"})
    return with_describe_pointer(res, "msm246")


mvsml_ann_gradient = mvsml_reproducing_kernel_eq_10_10


def cheatsheet():
    return "msm246: Gradient-descent weight change"
