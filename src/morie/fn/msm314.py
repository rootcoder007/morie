# morie.fn -- function file (rootcoder007/morie)

"""Functional regression with environment interaction -- re-export.

The generator emitted several modules for eq. (14.14) p.610 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm296 and this module re-exports it.
"""

from .msm296 import fregint

__all__ = ["fregint", "mvsml_convolutional_nn_eq_14_14"]

mvsml_convolutional_nn_eq_14_14 = fregint


def cheatsheet():
    return "msm314: Functional regression with environment interaction (see msm296)"
