# morie.fn -- function file (rootcoder007/morie)

"""Penalized functional regression fit -- re-export.

The generator emitted several modules for eq. (14.12) p.601 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm283 and this module re-exports it.
"""

from .msm283 import penfreg

__all__ = ["penfreg", "mvsml_convolutional_nn_eq_14_12"]

mvsml_convolutional_nn_eq_14_12 = penfreg


def cheatsheet():
    return "msm285: Penalized functional regression fit (see msm283)"
