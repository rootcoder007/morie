# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric links of the zero-altered Poisson forest.

Implements eq. (15.1) p.651 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_functional_regression_eq_15_1", "mvsml_zap_links"]


def mvsml_functional_regression_eq_15_1(mu_pred, theta_pred):
    """log(mu) = f_mu(x) and log(theta/(1-theta)) = f_theta(x)
    (eq. 15.1): under ZAP_RF and ZAPC_RF the links between covariates
    and response are general nonparametric functions estimated by two
    random forests rather than linear predictors. Keys: estimate."""
    f = _gp.zap_link(mu_pred, theta_pred)
    res = RichResult(payload={"estimate": f["mu"], "mu": f["mu"],
                              "theta": f["theta"],
                              "method": "ZAP nonparametric links (MVSML 2022 eq. 15.1)"})
    return with_describe_pointer(res, "msm323")


mvsml_zap_links = mvsml_functional_regression_eq_15_1


def cheatsheet():
    return "msm323: Nonparametric links of the zero-altered Poisson forest"


# compact alias per ledger/NAMING.md
mvsmlzaplinks = mvsml_zap_links
