# morie.fn -- function file (rootcoder007/morie)

"""Wolfe dual of the support vector classifier -- re-export.

The generator emitted several modules for eq. (9.44), (9.45) p.357 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm231 and this module re-exports it.
"""

from .msm231 import svmsdual

__all__ = ["svmsdual", "mvsml_ridge_lasso_elastic_eq_9_44"]

mvsml_ridge_lasso_elastic_eq_9_44 = svmsdual


def cheatsheet():
    return "msm233: Wolfe dual of the support vector classifier (see msm231)"
