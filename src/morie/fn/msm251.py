# morie.fn -- function file (rootcoder007/morie)
"""Output-layer weight update.

Implements eq. (10.13) p.411 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_13", "mvsml_ann_update_output"]


def mvsml_reproducing_kernel_eq_10_13(X, y, W, activations=None, eta=0.1, n_iter=1):
    """w_jk^(l)(t+1) = w_jk^(l)(t) + Delta w_jk^(l)
    = w_jk^(l)(t) + eta delta_ij V_ik^(h) (eq. 10.13): the adjustment
    is added to the current estimate to obtain the updated weight.
    Keys: estimate."""
    f = _gp.ann_train(X, y, W, eta=eta, n_iter=n_iter,
                      activations=activations)
    res = RichResult(payload={"estimate": f["loss"],
                              "W": f["W"], "loss": f["loss"],
                              "history": f["history"],
                              "method": "output weight update (MVSML 2022 eq. 10.13)"})
    return with_describe_pointer(res, "msm251")


mvsml_ann_update_output = mvsml_reproducing_kernel_eq_10_13


def cheatsheet():
    return "msm251: Output-layer weight update"
