# morie.fn -- function file (rootcoder007/morie)
"""Sum-of-squares loss.

Implements eq. (10.5) p.409 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_5", "mvsml_ann_sse"]


def mvsml_reproducing_kernel_eq_10_5(y_hat, y):
    """E = (1/2) sum_i sum_j (y-hat_ij - y_ij)^2 (eq. 10.5), the loss
    whose partial derivatives with respect to the weights drive
    backpropagation. Keys: estimate."""
    v = _gp.ann_sse(y_hat, y)
    res = RichResult(payload={"estimate": v, "sse": v,
                              "method": "SSE loss (MVSML 2022 eq. 10.5)"})
    return with_describe_pointer(res, "msm249")


mvsml_ann_sse = mvsml_reproducing_kernel_eq_10_5


def cheatsheet():
    return "msm249: Sum-of-squares loss"
