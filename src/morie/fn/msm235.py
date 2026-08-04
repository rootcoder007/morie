# morie.fn -- function file (rootcoder007/morie)

"""Support vector machine with a kernel -- re-export.

The generator emitted several modules for eq. (9.46), (9.47) p.360 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm234 and this module re-exports it.
"""

from .msm234 import ksvmdual

__all__ = ["ksvmdual", "mvsml_ridge_lasso_elastic_eq_9_47"]

mvsml_ridge_lasso_elastic_eq_9_47 = ksvmdual


def cheatsheet():
    return "msm235: Support vector machine with a kernel (see msm234)"
