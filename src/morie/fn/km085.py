# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.9: the Categorical Bias Score."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_cbs_variance"]


def kamath_ch6_cbs_variance(W, A, p_a, p_prior, ddof=0):
    """CBS = (1/|W|) sum_{w in W} Var_{a in A} log(p_a / p_prior).

    The non-binary generalisation of LPBS: instead of a difference
    between two groups it takes the VARIANCE of the prior-normalised
    log probabilities across all groups, then averages over template
    words. 0 means perfectly equal treatment; it can never be negative.
    ``p_a`` and ``p_prior`` are |W| x |A| matrices.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.9, printed
    p. 235.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_cbs_variance(["w1"], ["a1", "a2"],
    ...     [[0.5, 0.25]], [[0.25, 0.25]])
    >>> abs(out["estimate"] - (math.log(2.0) / 2) ** 2) < 1e-12
    True
    >>> kamath_ch6_cbs_variance(["w1"], ["a1", "a2"], [[0.5, 0.5]],
    ...                         [[0.25, 0.25]])["estimate"]
    0.0
    """
    words, attrs = list(W), list(A)
    if not words:
        raise ValueError("W is empty; a mean over no template words is "
                         "undefined.")
    if len(attrs) < 2:
        raise ValueError("A needs at least two social groups; the variance "
                         "of one value is not a bias score.")
    pa = np.atleast_2d(np.asarray(p_a, dtype=float))
    pp = np.atleast_2d(np.asarray(p_prior, dtype=float))
    if pa.shape != (len(words), len(attrs)):
        raise ValueError(
            f"p_a is {pa.shape} but |W| x |A| is "
            f"{(len(words), len(attrs))}.")
    if pp.shape != pa.shape:
        raise ValueError(f"p_prior is {pp.shape} but p_a is {pa.shape}.")
    if np.any(pa <= 0) or np.any(pp <= 0) or np.any(pa > 1) or np.any(pp > 1):
        raise ValueError("every probability must lie in (0, 1].")
    ddof = int(ddof)
    if len(attrs) - ddof <= 0:
        raise ValueError(f"ddof = {ddof} leaves no degrees of freedom.")
    logs = np.log(pa / pp)
    per_word = np.var(logs, axis=1, ddof=ddof)
    return RichResult(payload={
        "estimate": float(per_word.mean()),
        "per_word": [float(v) for v in per_word],
        "log_ratios": [[float(v) for v in row] for row in logs],
        "ddof": ddof, "n": len(words),
        "method": "Categorical Bias Score (Kamath Eq 6.9)"})


def cheatsheet():
    return "km085: mean over words of Var over groups of log(p/prior)"
