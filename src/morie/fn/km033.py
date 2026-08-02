# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.33: the denoising autoencoder loss."""

from . import _array_core as np

from ._richresult import RichResult
from .km021 import kamath_ch2_clm_loss

__all__ = ["kamath_ch2_dae_loss"]


def kamath_ch2_dae_loss(x, xhat):
    """L_DAE = -(1/|x|) sum log P(x_i | x_hat, x_<i): the CLM form
    conditioned on the corrupted input, so it delegates to Eq 2.21;
    what the conditioning changes is the model, not the loss.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.33, printed
    p. 55.

    Examples
    --------
    >>> import math
    >>> abs(kamath_ch2_dae_loss([0.5, 0.5], "noisy")["estimate"]
    ...     - math.log(2)) < 1e-12
    True
    """
    inner = kamath_ch2_clm_loss(x)
    return RichResult(payload={
        "estimate": inner["estimate"],
        "per_position": inner["per_position"], "n": inner["n"],
        "method": "Denoising autoencoder loss (Kamath Eq 2.33)"})


def cheatsheet():
    return "km033: DAE = CLM conditioned on the corrupted input"
