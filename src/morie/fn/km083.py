# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.7: CEAT's random-effects pooling of WEAT samples."""

import numpy as np

from ._richresult import RichResult
from .km082 import kamath_ch6_weat_effect_size

__all__ = ["kamath_ch6_ceat_random_effects"]


def kamath_ch6_ceat_random_effects(S_A1, S_A2, S_W1, S_W2, v, ddof=0):
    """CEAT = sum_i v_i WEAT_i / sum_i v_i.

    Contextualised embeddings give a DISTRIBUTION of effect sizes, one
    per sampled context; CEAT pools them with random-effects weights
    v_i (the inverse variances). Each WEAT_i is km082's, delegated.
    Equal weights reduce this to the plain mean, which the tests check.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.7, printed
    p. 234.

    Examples
    --------
    >>> A1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    >>> A2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    >>> W1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    >>> W2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    >>> out = kamath_ch6_ceat_random_effects(A1, A2, W1, W2, [1.0, 3.0])
    >>> out["estimate"], out["weat"]
    (2.0, [2.0, 2.0])
    """
    groups = [list(S_A1), list(S_A2), list(S_W1), list(S_W2)]
    N = len(groups[0])
    if N == 0:
        raise ValueError("no samples; a pooled effect over nothing is "
                         "undefined.")
    if any(len(g) != N for g in groups):
        raise ValueError(
            "S_A1, S_A2, S_W1 and S_W2 must hold the same number of "
            f"samples; got {[len(g) for g in groups]}.")
    w = np.atleast_1d(np.asarray(v, dtype=float))
    if w.size != N:
        raise ValueError(f"v has {w.size} weights for {N} samples.")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("every weight v_i must be finite and "
                         "non-negative.")
    if float(w.sum()) == 0:
        raise ValueError("the weights sum to 0; the pooled effect is "
                         "undefined.")
    eff = np.asarray([
        float(kamath_ch6_weat_effect_size(a1, a2, w1, w2, ddof=ddof)
              ["estimate"])
        for a1, a2, w1, w2 in zip(*groups)], dtype=float)
    return RichResult(payload={
        "estimate": float(np.sum(w * eff) / np.sum(w)),
        "weat": [float(x) for x in eff], "weights": [float(x) for x in w],
        "n": N,
        "method": "CEAT random-effects pooled WEAT (Kamath Eq 6.7)"})


def cheatsheet():
    return "km083: weighted mean of per-sample WEAT effect sizes"
