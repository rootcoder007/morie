# morie.fn -- function file (rootcoder007/morie)

"""Quadratic program under one linear inequality -- re-export.

The generator emitted several modules for eq. (9.15), (9.17), (9.18), (9.19), (9.20), (9.21), (9.22), (9.23), (9.24), (9.25), (9.26) p.346 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm188 and this module re-exports it.
"""

from .msm188 import qplincon

__all__ = ["qplincon", "mvsml_ridge_lasso_elastic_eq_9_23"]

mvsml_ridge_lasso_elastic_eq_9_23 = qplincon


def cheatsheet():
    return "msm195: Quadratic program under one linear inequality (see msm188)"
