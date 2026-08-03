# morie.fn -- function file (rootcoder007/morie)
"""Convolution over a receptive field.

Implements eq. (13.1) p.551 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_deep_learning_eq_13_1", "mvsml_cnn_convolve"]


def mvsml_deep_learning_eq_13_1(image, kernel, bias=0.0, stride=1):
    """z_i = w'x + b over the local receptive field (eq. 13.1): the
    filter slides across the image and takes a dot product with each
    local patch.  A 7x7x3 filter carries 148 parameters against the
    196,609 a fully connected layer would need, because the weights
    are shared across positions. Keys: estimate."""
    fm = _gp.conv2d(image, kernel, bias=bias, stride=stride)
    res = RichResult(payload={"estimate": fm[0][0],
                              "feature_map": fm,
                              "output_shape": (len(fm), len(fm[0])),
                              "method": "convolution (MVSML 2022 eq. 13.1)"})
    return with_describe_pointer(res, "msm259")


mvsml_cnn_convolve = mvsml_deep_learning_eq_13_1


def cheatsheet():
    return "msm259: Convolution over a receptive field"
