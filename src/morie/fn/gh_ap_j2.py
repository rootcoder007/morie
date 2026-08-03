# morie.fn -- function file (rootcoder007/morie)
"""CRM Laplace functional.

Implements Appendix J of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_crm_laplace"]


def ghosal_crm_laplace(f_val=1.0, gamma_shape=2.0, u_max=15.0,
                       n_grid=6000):
    """log E e^{-int f dM} = -int (1 - e^{-f u}) nu(du) (App J): for
    the gamma CRM nu(du) = a u^{-1} e^{-u} du this equals
    -a log(1 + f) exactly. Quadrature vs closed form.
    Keys: estimate."""
    a = gamma_shape
    num = 0.0
    for i in range(n_grid):
        u = (i + 0.5) * u_max / n_grid
        num += (1.0 - math.exp(-f_val * u)) * a / u \
            * math.exp(-u) * u_max / n_grid
    closed = a * math.log(1.0 + f_val)
    res = RichResult(payload={"estimate": num,
                              "closed_form": closed,
                              "gap": abs(num - closed),
                              "method": "CRM Laplace functional (GvdV 2017 App J)"})
    return with_describe_pointer(res, "gh_ap_j2")


def cheatsheet():
    return "gh_ap_j2: CRM Laplace functional"
