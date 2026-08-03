# morie.fn -- function file (rootcoder007/morie)
"""Predicted output of the network.

Implements eq. (10.8)-(10.9) p.409 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_9", "mvsml_ann_output"]


def mvsml_reproducing_kernel_eq_10_9(V_h, W_l, activation="identity"):
    """z_ij^(l) = sum_k w_jk^(l) V_ik^(h) (eq. 10.8) and
    y-hat_ij = g^(l)(z_ij^(l)) (eq. 10.9).  The output activation is
    the inverse-link that fixes the response type -- identity for
    continuous, logistic for binary, exponential for counts.
    Keys: estimate."""
    f = _gp.ann_forward(V_h, [W_l], [activation])
    res = RichResult(payload={"estimate": f["output"][0][0],
                              "z": f["nets"][0],
                              "y_hat": f["output"],
                              "method": "output-layer prediction (MVSML 2022 eq. 10.8-10.9)"})
    return with_describe_pointer(res, "msm248")


mvsml_ann_output = mvsml_reproducing_kernel_eq_10_9


def cheatsheet():
    return "msm248: Predicted output of the network"
