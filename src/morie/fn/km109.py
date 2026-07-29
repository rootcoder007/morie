# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.33: the worst-case perplexity leakage metric."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_perplexity_leakage"]


def _pp(source, w, name):
    if callable(source):
        v = float(source(w))
    else:
        if w not in source:
            raise ValueError(f"{name} has no perplexity for {w!r}.")
        v = float(source[w])
    if not np.isfinite(v) or v <= 0:
        raise ValueError(
            f"{name}({w!r}) = {v:.6g}; a perplexity must be finite and "
            "strictly positive.")
    return v


def kamath_ch6_perplexity_leakage(S_uniq, PP_public, PP_lm):
    """eps_l = max_{w in S_uniq} log[PP_public(w) / PP_lm(w)].

    The WORST sequence, not the average: how much less surprised the
    user-data model is by a unique sequence than a public model that
    never saw it. Above 0 means the model has memorised something the
    public model has not. ``PP_public`` and ``PP_lm`` are mappings or
    callables sequence -> perplexity.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.33, printed
    p. 259.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_perplexity_leakage(
    ...     ["w1", "w2"], {"w1": 10.0, "w2": 4.0},
    ...     {"w1": 5.0, "w2": 4.0})
    >>> abs(out["estimate"] - math.log(2.0)) < 1e-12, out["argmax"]
    (True, 'w1')
    """
    seqs = list(S_uniq)
    if not seqs:
        raise ValueError("S_uniq is empty; a maximum over no sequences is "
                         "undefined.")
    ratios = []
    for w in seqs:
        pub = _pp(PP_public, w, "PP_public")
        lm = _pp(PP_lm, w, "PP_lm")
        ratios.append(float(np.log(pub / lm)))
    arr = np.asarray(ratios, dtype=float)
    k = int(np.argmax(arr))
    return RichResult(payload={
        "estimate": float(arr[k]), "argmax": seqs[k],
        "per_sequence": ratios, "n_leaking": int(np.sum(arr > 0)),
        "n": len(seqs),
        "method": "worst-case perplexity leakage (Kamath Eq 6.33)"})


def cheatsheet():
    return "km109: max log(PP_public / PP_lm) over unique sequences"
