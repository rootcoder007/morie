# morie.fn -- function file (rootcoder007/morie)
"""Hidden-layer weight update and training loop.

Implements eq. (10.17) p.412 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter;
chapter 10 is Fundamentals of Artificial Neural Networks and Deep
Learning, and the canonical name below reflects that.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_reproducing_kernel_eq_10_17", "mvsml_ann_train"]


def mvsml_reproducing_kernel_eq_10_17(X, y, W, activations=None, eta=0.1, n_iter=500):
    """w_kp^(h)(t+1) = w_kp^(h)(t) + eta psi_ik x_ip (eq. 10.17),
    iterated with (10.13) over the feedforward/backward steps of the
    algorithm on p.412 until the loss stops decreasing.
    Keys: estimate."""
    f = _gp.ann_train(X, y, W, eta=eta, n_iter=n_iter,
                      activations=activations)
    res = RichResult(payload={"estimate": f["loss"],
                              "W": f["W"], "loss": f["loss"],
                              "history": f["history"],
                              "iterations": f["iterations"],
                              "output": f["output"],
                              "method": "backpropagation training (MVSML 2022 eq. 10.17)"})
    return with_describe_pointer(res, "msm255")


mvsml_ann_train = mvsml_reproducing_kernel_eq_10_17


def cheatsheet():
    return "msm255: Hidden-layer weight update and training loop"


# compact alias per ledger/NAMING.md
mvsmlanntrain = mvsml_ann_train
