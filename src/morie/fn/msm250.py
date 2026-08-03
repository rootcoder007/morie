# morie.fn -- function file (rootcoder007/morie)
"""Output-layer delta rule.

Implements eq. (10.12) p.411 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_12", "mvsml_ann_output_delta"]


def mvsml_reproducing_kernel_eq_10_12(X, y, W, activations=None, eta=0.1):
    """Delta w_jk^(l) = eta (y_ij - y-hat_ij) g^(l)'(z_ij^(l))
    V_ik^(h) = eta delta_ij V_ik^(h) (eq. 10.12), where
    delta_ij = (y_ij - y-hat_ij) g^(l)'(z_ij^(l)).  Keys: estimate."""
    g = _gp.ann_backprop_gradients(X, y, W, activations)
    last = g["gradients"][-1]
    upd = [[-eta * v for v in row] for row in last]
    res = RichResult(payload={"estimate": upd[0][0],
                              "delta_w": upd,
                              "gradient": last,
                              "method": "output delta rule (MVSML 2022 eq. 10.12)"})
    return with_describe_pointer(res, "msm250")


mvsml_ann_output_delta = mvsml_reproducing_kernel_eq_10_12


def cheatsheet():
    return "msm250: Output-layer delta rule"
