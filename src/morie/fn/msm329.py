# morie.fn -- function file (rootcoder007/morie)
"""ZAPC random-forest prediction.

Implements eq. (15.4) p.652 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_functional_regression_eq_15_4", "mvsml_zapc_predict"]


def mvsml_functional_regression_eq_15_4(theta_hat, mu_hat, threshold=0.5):
    """Y-hat = 0 when theta-hat > 0.5 and mu-hat otherwise
    (eq. 15.4): ZAPC_RF converts the probability to a zero rather than
    to a binary label, and the 0.5 threshold is used because it
    assumes no prior information. Keys: estimate."""
    v = _gp.zapc_predict(theta_hat, mu_hat, threshold=threshold)
    res = RichResult(payload={"estimate": v, "y_hat": v,
                              "is_zero": v == 0.0,
                              "method": "ZAPC_RF prediction (MVSML 2022 eq. 15.4)"})
    return with_describe_pointer(res, "msm329")


mvsml_zapc_predict = mvsml_functional_regression_eq_15_4


def cheatsheet():
    return "msm329: ZAPC random-forest prediction"
