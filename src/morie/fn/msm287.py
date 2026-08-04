# morie.fn -- function file (rootcoder007/morie)

"""Roughness penalty matrix -- re-export.

The generator emitted several modules for eq. (14.11) p.601 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm278 and this module re-exports it.
"""

from .msm278 import penmat

__all__ = ["penmat", "mvsml_convolutional_nn_eq_14_11"]

mvsml_convolutional_nn_eq_14_11 = penmat


def cheatsheet():
    return "msm287: Roughness penalty matrix (see msm278)"
