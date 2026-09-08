# morie.fn -- function file (rootcoder007/morie)
"""Bayesian multi-trait multi-environment model (BMTME).

Implements eq. (6.11) pp.195-196 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["bmtme_conditionals"]


def bmtme_conditionals(Y, Z1, Z2, G, Sigma_T, Sigma_E, R, b1=None, b2=None,
         **kw):
    """Y = 1_IJ mu' + X B + Z_1 b_1 + Z_2 b_2 + E (eq. 6.11): the
    multi-trait model extended with a trait x genotype x environment
    interaction, b_2 | Sigma_T, Sigma_E ~ MN(0, Sigma_E (x) G,
    Sigma_T).  Returns the two inverse-Wishart full conditionals of
    steps 5 and 6 on p.196, which are what distinguishes BMTME from
    eq. (6.9). Keys: estimate."""
    f = _gp.bmtme_conditionals(Y, Z1, Z2, G, Sigma_T, Sigma_E, R,
                               b1=b1, b2=b2, **kw)
    res = RichResult(payload={"estimate": f["scale_T"][0][0],
                              "nu_T_post": f["nu_T_post"],
                              "scale_T": f["scale_T"],
                              "nu_E_post": f["nu_E_post"],
                              "scale_E": f["scale_E"],
                              "method": "BMTME full conditionals (MVSML 2022 eq. 6.11)"})
    return with_describe_pointer(res, "msm076")


def cheatsheet():
    return "msm076: Bayesian multi-trait multi-environment model (BMTME)"
