# morie.fn -- function file (rootcoder007/morie)
"""Hidden-layer weight change.

Implements eq. (10.14)-(10.15) p.411 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_14", "mvsml_ann_hidden_gradient"]


def mvsml_reproducing_kernel_eq_10_14(X, y, W, activations=None, eta=0.1):
    """Delta w_kp^(h) = -eta dE/dw_kp^(h) (eq. 10.14), expanded by the
    chain rule through dz^(l)/dV^(h) = w_jk^(l),
    dV^(h)/dz^(h) = g^(h)'(z_ik^(h)) and dz^(h)/dw^(h) = x_ip
    (eq. 10.15). Keys: estimate."""
    g = _gp.ann_backprop_gradients(X, y, W, activations)
    first = g["gradients"][0]
    upd = [[-eta * v for v in row] for row in first]
    res = RichResult(payload={"estimate": upd[0][0],
                              "delta_w": upd, "gradient": first,
                              "method": "hidden-layer weight change (MVSML 2022 eq. 10.14-10.15)"})
    return with_describe_pointer(res, "msm253")


mvsml_ann_hidden_gradient = mvsml_reproducing_kernel_eq_10_14


def cheatsheet():
    return "msm253: Hidden-layer weight change"
