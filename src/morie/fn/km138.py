# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.10: SimVLM masked language modelling with image regions."""

import numpy as np

from ._richresult import RichResult
from .km022 import kamath_ch2_mlm_loss

__all__ = ["kamath_ch9_simvlm_mlm"]


def kamath_ch9_simvlm_mlm(theta, x, v, x_m):
    r"""L_MLM = -E log P_theta(x_m | x_{not m}, v).

    Arithmetically this is the Ch 2 MLM loss (Eq 2.22) with the image
    regions ``v`` added to the conditioning set, so the averaging of
    -log P over the masked positions is delegated to
    ``morie.fn.km022``; what is added here is the visual conditioning
    contract.

    ``theta`` is either the caller's model, a callable
    ``theta(x, v) -> probability of the true token at each position``,
    or ``None``, in which case ``x`` already holds those
    probabilities. ``x_m`` is the set of masked indices.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.10, printed
    p. 387; Wang et al. (2021).

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_simvlm_mlm(None, [0.5, 1.0, 0.25],
    ...                             [[0.0]], [0, 2])
    >>> abs(out["estimate"] - (math.log(2) + math.log(4)) / 2) < 1e-12
    True
    """
    if v is None:
        raise ValueError("v (the image regions) is required: Eq 9.10 "
                         "conditions on them; use km022 for text-only "
                         "MLM.")
    V = np.asarray(v, dtype=float)
    if V.size == 0:
        raise ValueError("the image-region features are empty.")
    if theta is None:
        probs = x
    elif callable(theta):
        probs = theta(x, v)
    else:
        raise ValueError("theta must be a callable model theta(x, v) "
                         "or None when x already holds the true-token "
                         "probabilities.")
    base = kamath_ch2_mlm_loss(probs, x_m)
    return RichResult(payload={
        "estimate": base["estimate"],
        "per_position": base["per_position"],
        "positions_scored": base["positions_scored"],
        "n_image_regions": int(V.shape[0]), "n": base["n"],
        "method": "SimVLM MLM loss with visual conditioning "
                  "(Kamath Eq 9.10; the MLM core in km022)"})


def cheatsheet():
    return "km138: km022's masked-token loss, conditioned on image regions"
