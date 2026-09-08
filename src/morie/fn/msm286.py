# morie.fn -- function file (rootcoder007/morie)

"""Penalized sum of squared errors -- re-export.

The generator emitted several modules for eq. (14.10) p.599 of
Montesinos Lopez, Montesinos Lopez & Crossa (2022), *Multivariate Statistical
Machine Learning Methods for Genomic Prediction*, Springer
(DOI 10.1007/978-3-030-89010-0).
All of them are the same method, so the implementation lives once in
morie.fn.msm277 and this module re-exports it.
"""

from .msm277 import pensse

__all__ = ["pensse", "mvsml_convolutional_nn_eq_14_10"]

mvsml_convolutional_nn_eq_14_10 = pensse


def cheatsheet():
    return "msm286: Penalized sum of squared errors (see msm277)"
