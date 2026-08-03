# morie.fn -- function file (rootcoder007/morie)
"""Net input of a hidden neuron.

Implements eq. (10.6)-(10.7) p.409 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_6", "mvsml_ann_hidden_net"]


def mvsml_reproducing_kernel_eq_10_6(X, W_h, activation="logistic"):
    """z_ik^(h) = sum_p w_kp^(h) x_ip (eq. 10.6) and
    V_ik^(h) = g^(h)(z_ik^(h)) (eq. 10.7).  The bias b_k^(h) is left
    out of (10.6) because it is carried by an extra input neuron fixed
    at 1. Keys: estimate."""
    f = _gp.ann_forward(X, [W_h], [activation])
    res = RichResult(payload={"estimate": f["nets"][0][0][0],
                              "z": f["nets"][0],
                              "V": f["output"],
                              "method": "hidden-layer net input and output (MVSML 2022 eq. 10.6-10.7)"})
    return with_describe_pointer(res, "msm247")


mvsml_ann_hidden_net = mvsml_reproducing_kernel_eq_10_6


def cheatsheet():
    return "msm247: Net input of a hidden neuron"
