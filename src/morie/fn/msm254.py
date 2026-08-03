# morie.fn -- function file (rootcoder007/morie)
"""Hidden-layer delta rule.

Implements eq. (10.16) p.412 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_16", "mvsml_ann_hidden_delta"]


def mvsml_reproducing_kernel_eq_10_16(X, y, W, activations=None, eta=0.1):
    """Delta w_kp^(h) = eta sum_j delta_ij w_jk^(l) g^(h)'(z_ik^(h))
    x_ip = eta psi_ik x_ip (eq. 10.16).  The sum runs over the output
    units because every hidden neuron is connected to all of them, so
    a change in one input-to-hidden weight moves every output.
    Keys: estimate."""
    g = _gp.ann_backprop_gradients(X, y, W, activations)
    first = g["gradients"][0]
    upd = [[-eta * v for v in row] for row in first]
    res = RichResult(payload={"estimate": upd[0][0],
                              "delta_w": upd,
                              "method": "hidden delta rule (MVSML 2022 eq. 10.16)"})
    return with_describe_pointer(res, "msm254")


mvsml_ann_hidden_delta = mvsml_reproducing_kernel_eq_10_16


def cheatsheet():
    return "msm254: Hidden-layer delta rule"
