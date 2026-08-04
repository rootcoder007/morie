# morie.fn -- function file (rootcoder007/morie)

"""Support vector classifier (soft margin) -- re-export.

The generator emitted several modules for eq. (9.34), (9.35), (9.36), (9.37) p.354 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm218 and this module re-exports it.
"""

from .msm218 import softsvm

__all__ = ["softsvm", "mvsml_ridge_lasso_elastic_eq_9_37"]

mvsml_ridge_lasso_elastic_eq_9_37 = softsvm


def cheatsheet():
    return "msm222: Support vector classifier (soft margin) (see msm218)"
