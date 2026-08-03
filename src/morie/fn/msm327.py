# morie.fn -- function file (rootcoder007/morie)
"""ZAP random-forest prediction.

Implements eq. (15.3) p.652 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_functional_regression_eq_15_3", "mvsml_zap_predict"]


def mvsml_functional_regression_eq_15_3(theta_hat, mu_hat):
    """Y-hat = (1 - theta-hat) exp(-mu-hat)/(1 - exp(-mu-hat))
    (eq. 15.3): under ZAP_RF the prediction is the mean of the
    zero-altered Poisson model. Keys: estimate."""
    v = _gp.zap_predict(theta_hat, mu_hat)
    res = RichResult(payload={"estimate": v, "y_hat": v,
                              "method": "ZAP_RF prediction (MVSML 2022 eq. 15.3)"})
    return with_describe_pointer(res, "msm327")


mvsml_zap_predict = mvsml_functional_regression_eq_15_3


def cheatsheet():
    return "msm327: ZAP random-forest prediction"
