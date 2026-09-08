# morie.fn -- function file (rootcoder007/morie)

"""Functional regression with environment effects -- re-export.

The generator emitted several modules for eq. (14.13) p.607 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm293 and this module re-exports it.
"""

from .msm293 import fregenv

__all__ = ["fregenv", "mvsml_convolutional_nn_eq_14_13"]

mvsml_convolutional_nn_eq_14_13 = fregenv


def cheatsheet():
    return "msm295: Functional regression with environment effects (see msm293)"
