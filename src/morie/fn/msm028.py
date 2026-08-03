# morie.fn -- function file (rootcoder007/morie)
"""Multi-trait model with extra fixed effects.

Implements eq. (5.5a) p.153 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_linear_mixed_models_eq_5_5a"]


def mvsml_linear_mixed_models_eq_5_5a(Y, Z, G, Sigma_T, R_T, X=None):
    """Y = (1_IJ (x) I_nT) mu + X beta + Z b + eps (eq. 5.5a): the
    multi-trait model of eq. (5.5) extended with a fixed-effects term
    X beta. Keys: estimate."""
    f = _gp.multitrait_model(Y, Z, G, Sigma_T, R_T, X=X)
    res = RichResult(payload={"estimate": f["mu"][0], "mu": f["mu"],
                              "beta": f["beta"], "b": f["b"],
                              "method": "multi-trait LMM with fixed effects (MVSML 2022 eq. 5.5a)"})
    return with_describe_pointer(res, "msm028")


def cheatsheet():
    return "msm028: Multi-trait model with extra fixed effects"
