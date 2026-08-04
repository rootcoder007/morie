# morie.fn -- function file (rootcoder007/morie)

"""Karush-Kuhn-Tucker conditions of the support vector classifier -- re-export.

The generator emitted several modules for eq. (9.38), (9.39), (9.40), (9.41), (9.42), (9.43) p.356 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm223 and this module re-exports it.
"""

from .msm223 import svmkkt

__all__ = ["svmkkt", "mvsml_ridge_lasso_elastic_eq_9_39"]

mvsml_ridge_lasso_elastic_eq_9_39 = svmkkt


def cheatsheet():
    return "msm224: Karush-Kuhn-Tucker conditions of the support vector classifier (see msm223)"
