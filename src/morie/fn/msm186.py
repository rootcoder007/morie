# morie.fn -- function file (rootcoder007/morie)

"""Wolfe dual of a constrained program -- re-export.

The generator emitted several modules for eq. (9.9), (9.12), (9.13), (9.14) p.346 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm184 and this module re-exports it.
"""

from .msm184 import wolfedual

__all__ = ["wolfedual", "mvsml_ridge_lasso_elastic_eq_9_13"]

mvsml_ridge_lasso_elastic_eq_9_13 = wolfedual


def cheatsheet():
    return "msm186: Wolfe dual of a constrained program (see msm184)"
