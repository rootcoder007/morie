# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.31: ProPILE's PII likelihood metric."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_pii_likelihood"]


def kamath_ch6_pii_likelihood(a_m, A, x, L_q, L_r):
    """P_r(a_m | A_without_m) = prod_{r=1..L_r}
    p(a_{m,r} | x_1,...,x_{L_q + r - 1}).

    The likelihood that the model reproduces a target piece of PII
    token by token, each token conditioned on the query AND everything
    of the answer emitted so far -- which is why the context length
    grows as L_q + r - 1. ``a_m`` holds those L_r per-token
    probabilities, ``x`` the L_q query tokens, ``A`` the other PII the
    prompt supplies. The log likelihood is returned too, since the
    product underflows for long answers.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.31, printed
    p. 258.

    Examples
    --------
    >>> out = kamath_ch6_pii_likelihood([0.5, 0.25], ["name"],
    ...                                 ["contact", "John"], 2, 2)
    >>> out["estimate"], out["context_lengths"]
    (0.125, [2, 3])
    """
    p = np.atleast_1d(np.asarray(a_m, dtype=float))
    q = list(x)
    Lq, Lr = int(L_q), int(L_r)
    if Lr < 1:
        raise ValueError("L_r must be at least 1; an empty answer has no "
                         "likelihood.")
    if p.size != Lr:
        raise ValueError(
            f"a_m holds {p.size} token probabilities but L_r = {Lr}.")
    if len(q) != Lq:
        raise ValueError(
            f"x holds {len(q)} query tokens but L_q = {Lq}.")
    if Lq < 1:
        raise ValueError("L_q must be at least 1; the model must be "
                         "prompted with something.")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("every token probability must lie in (0, 1].")
    return RichResult(payload={
        "estimate": float(np.prod(p)),
        "log_likelihood": float(np.sum(np.log(p))),
        "per_token": [float(v) for v in p],
        "context_lengths": [Lq + r - 1 for r in range(1, Lr + 1)],
        "n_other_pii": len(list(A)), "n": Lr,
        "method": "ProPILE PII likelihood (Kamath Eq 6.31)"})


def cheatsheet():
    return "km107: product of the target PII's per-token probabilities"
