# morie.fn -- function file (rootcoder007/morie)
"""Design matrix of the functional regression.

Implements eq. (14.9) pp.581-582 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_convolutional_nn_eq_14_9", "mvsml_fda_design"]


def mvsml_convolutional_nn_eq_14_9(t, X_curves, L1=3, L2=5, kind="fourier"):
    """X* = [1_n  X] with X = X** Psi (Psi'Psi)^-1 Q' (eq. 14.9),
    where Q collects the inner products int phi_l(t) psi_o(t) dt.
    Each row is x_i = Q c-hat_i. Keys: estimate."""
    d = _gp.fda_design_matrix(t, X_curves, L1, L2, kind=kind)
    res = RichResult(payload={"estimate": d["X_star"][0][0],
                              "X_star": d["X_star"], "Q": d["Q"],
                              "method": "functional design matrix (MVSML 2022 eq. 14.9)"})
    return with_describe_pointer(res, "msm272")


mvsml_fda_design = mvsml_convolutional_nn_eq_14_9


def cheatsheet():
    return "msm272: Design matrix of the functional regression"
