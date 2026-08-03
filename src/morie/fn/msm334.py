# morie.fn -- function file (rootcoder007/morie)
"""Expected prediction error of the OLS fit.

Implements sec. 3.5 p.80 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the auto-generated stub name carries a topic label that does not
match the book's chapter title; the chapter and equation numbers do
match, and the PDF is the authority followed here.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_preprocessing_eq_2_22"]


def mvsml_preprocessing_eq_2_22(sigma2, x_star, eigenvalues):
    """EPE(x_o) = sigma2 (1 + sum_j (x*_oj)^2 / lambda_j) with
    x* = Gamma'x_o and lambda_j the eigenvalues of X'X (p.80): nearly
    dependent features drive some lambda_j toward zero and blow the
    prediction error up. Keys: estimate."""
    v = _gp.expected_prediction_error(sigma2, x_star, eigenvalues)
    res = RichResult(payload={"estimate": v,
                              "irreducible": float(sigma2),
                              "variance_inflation": v / float(sigma2),
                              "method": "expected prediction error (MVSML 2022 p.80)"})
    return with_describe_pointer(res, "msm334")


def cheatsheet():
    return "msm334: Expected prediction error of the OLS fit"
