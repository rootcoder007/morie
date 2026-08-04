# morie.fn -- function file (rootcoder007/morie)

"""Maximum margin classifier -- re-export.

The generator emitted several modules for eq. (9.6), (9.7), (9.8) p.344 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm175 and this module re-exports it.
"""

from .msm175 import hardsvm

__all__ = ["hardsvm", "mvsml_ridge_lasso_elastic_eq_9_6"]

mvsml_ridge_lasso_elastic_eq_9_6 = hardsvm


def cheatsheet():
    return "msm180: Maximum margin classifier (see msm175)"
