# morie.fn -- function file (rootcoder007/morie)
"""Statistical learning model, systematic plus random part.

Implements eq. (1.1) p.8 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_general_eq_1_1"]


def mvsml_general_eq_1_1(x, f=None, noise=None):
    """y_i = f(x_i) + eps_i, i = 1..n (eq. 1.1): the systematic part
    f is determined by the predictors, eps has mean zero.
    Keys: estimate."""
    xs = _gp._flat(x)
    if f is None:
        f = lambda v: v
    sys_part = [float(f(v)) for v in xs]
    eps = [0.0] * len(xs) if noise is None else _gp._flat(noise)
    y = [a + b for a, b in zip(sys_part, eps)]
    res = RichResult(payload={"estimate": y[0], "y": y,
                              "systematic": sys_part,
                              "mean_error": sum(eps) / len(eps),
                              "method": "model = systematic + random (MVSML 2022 eq. 1.1)"})
    return with_describe_pointer(res, "msm001")


def cheatsheet():
    return "msm001: Statistical learning model, systematic plus random part"
