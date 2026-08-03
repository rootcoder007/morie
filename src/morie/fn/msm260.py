# morie.fn -- function file (rootcoder007/morie)
"""Activation map of a convolution layer.

Implements eq. (13.2) p.551 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).

Note: the stub name carries a topic label from another chapter; the
canonical name below reflects the chapter this equation is actually in.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_deep_learning_eq_13_2", "mvsml_cnn_activation_map"]


def mvsml_deep_learning_eq_13_2(image, kernel, bias=0.0, stride=1,
         activation="relu"):
    """Applying the activation to each net input of eq. (13.1) gives
    the feature (activation) map of eq. (13.2).  Every node of the map
    detects the same feature at a different position, which is what
    makes a CNN translationally invariant. Keys: estimate."""
    fm = _gp.conv2d(image, kernel, bias=bias, stride=stride,
                    activation=activation)
    res = RichResult(payload={"estimate": fm[0][0],
                              "activation_map": fm,
                              "output_shape": (len(fm), len(fm[0])),
                              "method": "activation map (MVSML 2022 eq. 13.2)"})
    return with_describe_pointer(res, "msm260")


mvsml_cnn_activation_map = mvsml_deep_learning_eq_13_2


def cheatsheet():
    return "msm260: Activation map of a convolution layer"
