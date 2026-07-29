# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.1: the multimodal modality encoder."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_modality_encoder"]


def kamath_ch9_modality_encoder(I_X, ME_X):
    r"""F_X = ME_X(I_X): encode a modality's input into features.

    Eq 9.1 is a composition, not a closed form, so ``ME_X`` is the
    caller's encoder and the contract is enforced here: it must be
    callable, and it must return a finite real feature array. The
    reported ``estimate`` is the L2 norm of the features -- the one
    scalar summary of an encoder output that is not a re-average of
    the input.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.1, printed
    p. 378.

    Examples
    --------
    >>> out = kamath_ch9_modality_encoder([3.0, 4.0],
    ...                                   lambda z: [z[0], z[1], 0.0])
    >>> out["estimate"]
    5.0
    >>> out["features"]
    [3.0, 4.0, 0.0]
    """
    if not callable(ME_X):
        raise ValueError("ME_X must be a callable modality encoder.")
    F = np.asarray(ME_X(I_X), dtype=float)
    if F.size == 0:
        raise ValueError("the modality encoder returned no features.")
    if not np.all(np.isfinite(F)):
        raise ValueError("the modality encoder returned non-finite "
                         "features.")
    return RichResult(payload={
        "estimate": float(np.linalg.norm(F)),
        "features": [float(v) for v in F.ravel()],
        "shape": list(F.shape), "n": int(F.size),
        "method": "modality encoder F_X = ME_X(I_X) (Kamath Eq 9.1)"})


def cheatsheet():
    return "km129: runs the caller's modality encoder, checks its output"
