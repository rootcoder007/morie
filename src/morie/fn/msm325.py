# morie.fn -- function file (rootcoder007/morie)
"""Zero-truncated Poisson splitting criterion.

Implements eq. (15.2) p.651 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_functional_regression_eq_15_2", "mvsml_zero_truncated_poisson_loglik"]


def mvsml_functional_regression_eq_15_2(y_positive, mu=None, x=None):
    """LL+ = -N+ log(1 - exp(-mu)) + log(mu) sum_i Y_i+ - N+ mu
    - sum_i log(Y_i+!) (eq. 15.2), the zero-truncated Poisson
    log-likelihood used as the splitting criterion in the truncated
    part of the forest.  With ``x`` the best split is searched, taking
    the one that maximizes LL+(left) + LL+(right) (p.652).
    Keys: estimate."""
    if mu is None:
        mu = _gp.zero_truncated_poisson_mle(y_positive)
    ll = _gp.zero_truncated_poisson_loglik(y_positive, mu)
    split = _gp.zap_best_split(y_positive, x) if x is not None \
        else None
    res = RichResult(payload={"estimate": ll, "loglik": ll,
                              "mu": mu, "split": split,
                              "method": "zero-truncated Poisson criterion (MVSML 2022 eq. 15.2)"})
    return with_describe_pointer(res, "msm325")


mvsml_zero_truncated_poisson_loglik = mvsml_functional_regression_eq_15_2


def cheatsheet():
    return "msm325: Zero-truncated Poisson splitting criterion"
