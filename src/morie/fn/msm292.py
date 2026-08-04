# morie.fn -- function file (rootcoder007/morie)

"""Functional linear model with scalar response -- re-export.

The generator emitted several modules for eq. (14.1) p.579 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm261 and this module re-exports it.
"""

from .msm261 import flmint

__all__ = ["flmint", "mvsml_convolutional_nn_eq_14_1"]

mvsml_convolutional_nn_eq_14_1 = flmint


def cheatsheet():
    return "msm292: Functional linear model with scalar response (see msm261)"
