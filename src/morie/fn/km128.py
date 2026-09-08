# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.16: the unbiased pass@k estimator."""

from ._richresult import RichResult

__all__ = ["kamath_ch8_pass_at_k"]


def kamath_ch8_pass_at_k(n, c, k):
    r"""pass@k = 1 - C(n-c, k) / C(n, k).

    ``n`` samples were generated per problem, ``c`` of them passed the
    unit tests, and ``k`` are drawn. The ratio is evaluated as the
    product prod_{i<k} (n-c-i)/(n-i), which never forms a large
    binomial and is exact in floating point for the sizes the book
    uses (n = 200, k <= 100).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.16, printed
    p. 328; Kulal et al. (2019); Chen et al. (2021).

    Examples
    --------
    >>> out = kamath_ch8_pass_at_k(5, 2, 2)
    >>> round(out["estimate"], 12)     # 1 - C(3,2)/C(5,2) = 1 - 3/10
    0.7
    >>> kamath_ch8_pass_at_k(5, 0, 2)["estimate"]
    0.0
    """
    n_i, c_i, k_i = int(n), int(c), int(k)
    if n_i != n or c_i != c or k_i != k:
        raise ValueError("n, c and k are sample counts and must be "
                         "integers.")
    if n_i < 1:
        raise ValueError("at least one sample must be generated.")
    if not (0 <= c_i <= n_i):
        raise ValueError(f"c = {c_i} must lie in [0, n] with n = {n_i}.")
    if not (1 <= k_i <= n_i):
        raise ValueError(f"k = {k_i} must lie in [1, n] with n = {n_i}.")
    if n_i - c_i < k_i:
        ratio = 0.0
    else:
        ratio = 1.0
        for i in range(k_i):
            ratio *= (n_i - c_i - i) / (n_i - i)
    return RichResult(payload={
        "estimate": 1.0 - ratio, "fail_probability": ratio,
        "n_samples": n_i, "n_correct": c_i, "k": k_i, "n": n_i,
        "method": "unbiased pass@k (Kamath Eq 8.16)"})


def cheatsheet():
    return "km128: 1 - P(all k drawn samples fail), hypergeometric"
