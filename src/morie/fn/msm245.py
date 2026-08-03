# morie.fn -- function file (rootcoder007/morie)
"""Feedforward network output.

Implements eq. (10.1)-(10.3) p.385 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_4", "mvsml_ann_forward"]


def mvsml_reproducing_kernel_eq_10_4(X, W, activations=None):
    """V_1j = g_1(sum_i w_ji^(1) x_i), V_2k = g_2(sum_j w_kj^(2) V_1j)
    and y_l = g_3(sum_k w_lk^(3) V_2k) (eq. 10.1-10.3): the analytical
    form of a network with d inputs, M_1 and M_2 hidden units and O
    outputs.  A bias is represented by an extra unit fixed at 1.
    Keys: estimate."""
    f = _gp.ann_forward(X, W, activations)
    res = RichResult(payload={"estimate": f["output"][0][0],
                              "output": f["output"],
                              "layers": f["layers"],
                              "nets": f["nets"],
                              "method": "feedforward pass (MVSML 2022 eq. 10.1-10.3)"})
    return with_describe_pointer(res, "msm245")


mvsml_ann_forward = mvsml_reproducing_kernel_eq_10_4


def cheatsheet():
    return "msm245: Feedforward network output"
