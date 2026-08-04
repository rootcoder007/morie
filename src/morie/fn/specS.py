# morie.fn -- function file (rootcoder007/morie)
"""Speculative decoding: acceptance rate and expected token yield."""

from math import fsum

from ._richresult import RichResult
from ._spx import vec

__all__ = [
    "speculative_decoding",
    "specdec",
]


def speculative_decoding(draft, target, gamma=4):
    """Acceptance rate and expected tokens per target call.

    NOT IN SCHABENBERGER & GOTWAY -- this is language-model inference, not
    spatial statistics; the module shares only the ``spec`` prefix. The
    source is Leviathan, Y., Kalman, M. & Matias, Y. (2023), "Fast
    inference from transformers via speculative decoding", ICML. Named
    from the general literature; NOT verified against a PDF in this
    corpus.

    A small draft model proposes gamma tokens; the large target model
    verifies them in ONE forward pass, accepting token i with probability
    min(1, p_i/q_i) where q is the draft distribution and p the target's.
    The per-token acceptance rate is therefore

        alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q),

    the total-variation distance complement, and the expected number of
    tokens produced per target call is the truncated geometric sum

        E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha),

    capped at gamma+1 because a rejected token is resampled from the
    residual distribution and still counts. At alpha = 1 the formula is
    the removable singularity gamma+1, handled explicitly rather than by
    dividing by zero.

    This function is DETERMINISTIC: it returns the acceptance rate and the
    expectation, not a sampled run. Sampling the accept/reject coin would
    make it impossible to compare across language arms, and the quantity
    anyone actually wants from this algorithm is the expected speedup.

    Parameters
    ----------
    draft, target : array-like
        Probability vectors over the same vocabulary; each must be
        non-negative and sum to 1 within 1e-9.
    gamma : int
        Tokens proposed per target call.

    Returns
    -------
    RichResult
        ``alpha``, ``tv_distance``, ``expected_tokens``, ``gamma``,
        ``max_tokens``, ``n``, ``method``.
    """
    q = vec(draft, "draft")
    p = vec(target, "target")
    if len(q) != len(p):
        raise ValueError("`draft` and `target` must cover the same vocabulary")
    if len(q) < 2:
        raise ValueError("a vocabulary of at least 2 tokens is needed")
    if any(t < 0 for t in q) or any(t < 0 for t in p):
        raise ValueError("probabilities must be non-negative")
    if abs(fsum(q) - 1.0) > 1e-9:
        raise ValueError("`draft` must sum to 1 (got %.12g)" % fsum(q))
    if abs(fsum(p) - 1.0) > 1e-9:
        raise ValueError("`target` must sum to 1 (got %.12g)" % fsum(p))
    g = int(gamma)
    if g < 1:
        raise ValueError("`gamma` must be at least 1")

    alpha = fsum([min(p[i], q[i]) for i in range(len(p))])
    tv = 1.0 - alpha
    if tv <= 1e-15:
        expect = float(g + 1)
    else:
        expect = (1.0 - alpha ** (g + 1)) / (1.0 - alpha)

    return RichResult(payload={
        "alpha": alpha,
        "tv_distance": tv,
        "expected_tokens": expect,
        "gamma": float(g),
        "max_tokens": float(g + 1),
        "deterministic_expectation_not_a_sampled_run": True,
        "n": len(p),
        "method": ("Speculative decoding acceptance rate and expected "
                   "token yield (Leviathan, Kalman & Matias 2023); NOT in "
                   "Schabenberger & Gotway"),
    })


def cheatsheet():
    return "specS: speculative decoding acceptance rate and token yield"


# compact alias per ledger/NAMING.md
specdec = speculative_decoding
