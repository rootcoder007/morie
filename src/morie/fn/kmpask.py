# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pass@k for code generation (HumanEval)."""

from math import comb

from ._richresult import RichResult

__all__ = ["kamath_pass_at_k"]


def kamath_pass_at_k(n, c, k):
    """pass@k = 1 - C(n - c, k) / C(n, k).

    The unbiased estimator: of ``n`` sampled programs ``c`` pass, and
    the value is the probability that a random draw of ``k`` contains
    at least one that passes. Computed as a product of
    (1 - c/(n - i)) terms rather than a ratio of binomials, which
    overflows for the n = 200 samples HumanEval actually uses.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, pass@k (Chen et al.
    2021).

    Examples
    --------
    >>> kamath_pass_at_k(10, 0, 1)["estimate"]
    0.0
    >>> kamath_pass_at_k(10, 10, 3)["estimate"]
    1.0
    >>> out = kamath_pass_at_k(4, 1, 2)
    >>> abs(out["estimate"] - (1 - comb(3, 2) / comb(4, 2))) < 1e-12
    True
    >>> round(out["estimate"], 10)
    0.5
    """
    n, c, k = int(n), int(c), int(k)
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    if not 0 <= c <= n:
        raise ValueError(f"c must lie in [0, {n}]; got {c}.")
    if not 1 <= k <= n:
        raise ValueError(
            f"k must lie in [1, {n}]; drawing {k} of {n} samples is not "
            "defined.")
    if n - c < k:
        value = 1.0
    else:
        # prod_{i=0}^{k-1} (n - c - i) / (n - i) == C(n-c,k)/C(n,k)
        fail = 1.0
        for i in range(k):
            fail *= (n - c - i) / (n - i)
        value = 1.0 - fail
    return RichResult(payload={
        "estimate": value, "pass_at_k": value,
        "n_samples": n, "n_correct": c, "k": k,
        "empirical_rate": c / n, "n": n,
        "method": "pass@k = 1 - C(n-c,k)/C(n,k) (unbiased estimator)"})


def cheatsheet():
    return "kmpask: 1 - C(n-c,k)/C(n,k) via an overflow-free product"


# compact alias per ledger/NAMING.md
kamathpassatk = kamath_pass_at_k
