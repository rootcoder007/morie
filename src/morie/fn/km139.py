# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.11: the SimVLM PrefixLM objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_simvlm_prefixlm"]


def kamath_ch9_simvlm_prefixlm(theta, x, T_p):
    r"""L_PrefixLM = -E_x log P_theta(x_{>=T_p} | x_{<T_p}).

    The suffix log-probability is a JOINT over the remaining tokens,
    so the per-sequence loss is the SUM of -log p over positions
    ``T_p ...`` (not their mean); the expectation over the data D is
    the mean across sequences.

    ``x`` holds the model's probability of the true token at each
    position -- one sequence (1-D) or a batch (2-D). ``theta`` may be
    a callable ``theta(x) -> those probabilities``, or ``None``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.11, printed
    p. 387.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_simvlm_prefixlm(None, [0.5, 0.5, 0.25], 1)
    >>> abs(out["estimate"] - math.log(8)) < 1e-12
    True
    """
    if theta is not None:
        if not callable(theta):
            raise ValueError("theta must be a callable theta(x) or "
                             "None when x already holds the true-token "
                             "probabilities.")
        x = theta(x)
    P = np.atleast_2d(np.asarray(x, dtype=float))
    if P.size == 0:
        raise ValueError("no token probabilities were given.")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("token probabilities must lie in [0, 1].")
    tp = int(T_p)
    if tp < 0 or tp >= P.shape[1]:
        raise ValueError(
            f"the prefix length {tp} leaves no suffix in a sequence of "
            f"{P.shape[1]} tokens.")
    with np.errstate(divide="ignore"):
        per_seq = -np.log(P[:, tp:]).sum(axis=1)
    return RichResult(payload={
        "estimate": float(per_seq.mean()),
        "per_sequence": [float(v) for v in per_seq],
        "prefix_length": tp, "n_suffix_tokens": int(P.shape[1] - tp),
        "n": int(P.shape[0]),
        "method": "PrefixLM suffix negative log-likelihood "
                  "(Kamath Eq 9.11)"})


def cheatsheet():
    return "km139: summed -log p over the suffix, averaged over sequences"
