# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LayerNorm -- per-token normalization.

Same method as :mod:`morie.fn.grln`: Ba, J. L., Kiros, J. R. and
Hinton, G. E. (2016), "Layer Normalization", arXiv:1607.06450,
Section 3 -- statistics (mean, variance) are computed over the hidden
units of a single training case, then a learned per-unit gain and bias
are applied: y = g * (x - mu) / sqrt(sigma^2 + eps) + b, with eps
inside the square root.

There is exactly one implementation: this module delegates to
:func:`morie.fn.grln.geron_layer_normalization`. A second copy would
agree with the first at 1e-9 forever and be indistinguishable from
correct work while doubling the surface under a name that reads right.

Source: fetched-wave3/ba-kiros-hinton-2016-layer-normalization-
arxiv1607.06450.pdf (Section 3, Eqs 2-4 region); identity of the
delegate verified against that definition (statistics per instance,
eps inside the root, affine after normalisation).
"""

from .grln import geron_layer_normalization as _impl

__all__ = ["layrnm", "layer_norm"]


def layrnm(X, gamma=1.0, beta=0.0, eps=1e-5):
    """LayerNorm (Ba, Kiros and Hinton 2016, arXiv:1607.06450, Sec 3).

    Delegates to :func:`morie.fn.grln.geron_layer_normalization`;
    see that function for parameters and payload.
    """
    return _impl(X, gamma=gamma, beta=beta, eps=eps)


layer_norm = layrnm


def cheatsheet():
    return "layrnm: LayerNorm (Ba et al. 2016, arXiv:1607.06450) -- alias of grln.geron_layer_normalization"
