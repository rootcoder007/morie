# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.13: the image-text matching (ITM) binary loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_itm_loss"]


def kamath_ch9_itm_loss(theta, v, t, y):
    r"""L_ITM = -E[ y log s(v,t) + (1-y) log(1 - s(v,t)) ].

    ``theta`` is either a callable similarity head ``theta(v, t) -> s``
    or the vector of similarity scores s_theta(v, t) itself; ``y`` is
    the 0/1 matched-pair label. The expectation is taken as the mean
    over the pairs supplied.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.13, printed
    p. 388.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_itm_loss([0.9, 0.2], None, None, [1, 0])
    >>> abs(out["estimate"]
    ...     - (-math.log(0.9) - math.log(0.8)) / 2) < 1e-12
    True
    """
    s = theta(v, t) if callable(theta) else theta
    s = np.atleast_1d(np.asarray(s, dtype=float))
    lab = np.atleast_1d(np.asarray(y, dtype=float))
    if s.size == 0:
        raise ValueError("no image-text pairs were scored.")
    if s.shape != lab.shape:
        raise ValueError(
            f"{lab.size} labels for {s.size} similarity scores.")
    if np.any((s < 0) | (s > 1)):
        raise ValueError("s_theta(v, t) is a probability and must lie "
                         "in [0, 1]; apply the sigmoid first.")
    if not np.all((lab == 0) | (lab == 1)):
        raise ValueError("ITM labels must be 0 (mismatched) or 1 "
                         "(matched).")
    # score of the CORRECT label per pair -- written this way so a
    # confident-and-right score of exactly 1 gives 0, not 0 * -inf.
    correct = np.where(lab == 1, s, 1.0 - s)
    with np.errstate(divide="ignore"):
        per = -np.log(correct)
    return RichResult(payload={
        "estimate": float(per.mean()),
        "per_pair": [float(u) for u in per],
        "scores": [float(u) for u in s], "n": int(s.size),
        "method": "image-text matching binary loss (Kamath Eq 9.13)"})


def cheatsheet():
    return "km141: binary cross-entropy on image-text match scores"
