# morie.fn -- function file (rootcoder007/morie)

"""Hyperplane definition and side -- re-export.

The generator emitted several modules for eq. (9.1), (9.2), (9.3) p.339 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm161 and this module re-exports it.
"""

from .msm161 import hyperpl

__all__ = ["hyperpl", "mvsml_ridge_lasso_elastic_eq_9_3"]

mvsml_ridge_lasso_elastic_eq_9_3 = hyperpl


def cheatsheet():
    return "msm170: Hyperplane definition and side (see msm161)"
