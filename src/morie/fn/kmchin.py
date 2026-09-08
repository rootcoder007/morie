# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 1: Chinchilla compute-optimal model and data size."""

import math

from ._richresult import RichResult

__all__ = ["kamath_chinchilla_compute_optimal"]


def kamath_chinchilla_compute_optimal(compute_budget, alpha=0.5,
                                      beta=0.5, tokens_per_param=20.0,
                                      flops_per_token_param=6.0):
    r"""Split a FLOP budget as C = 6 N D with D / N = 20.

    Hoffmann et al.'s finding is that N and D should scale together,
    N ~ C^alpha and D ~ C^beta with alpha = beta = 0.5, which the
    accounting identity C = 6ND forces to satisfy alpha + beta = 1 --
    so a pair that does not is refused rather than quietly ignored.
    Given that, the token/parameter ratio pins the split:
    N = sqrt(C / (6 * ratio)) and D = ratio * N.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 1, Compute-Optimal
    Scaling; Hoffmann et al. (2022).

    Examples
    --------
    >>> out = kamath_chinchilla_compute_optimal(1.2e10)
    >>> out["N_opt"], out["D_opt"]        # 6 * 1e4 * 2e5 = 1.2e10
    (10000.0, 200000.0)
    """
    C = float(compute_budget)
    ratio = float(tokens_per_param)
    kflop = float(flops_per_token_param)
    if C <= 0:
        raise ValueError("the compute budget must be positive.")
    if ratio <= 0:
        raise ValueError("the tokens-per-parameter ratio must be "
                         "positive.")
    if kflop <= 0:
        raise ValueError("the FLOPs-per-token-per-parameter constant "
                         "must be positive.")
    if abs(float(alpha) + float(beta) - 1.0) > 1e-9:
        raise ValueError(
            f"alpha + beta = {float(alpha) + float(beta)}, but C = "
            "6ND forces the exponents to sum to 1.")
    N = math.sqrt(C / (kflop * ratio))
    D = ratio * N
    return RichResult(payload={
        "estimate": N, "N_opt": N, "D_opt": D,
        "tokens_per_param": ratio, "compute_budget": C,
        "compute_check": kflop * N * D, "alpha": float(alpha),
        "beta": float(beta), "n": 1,
        "method": "Chinchilla compute-optimal split (Kamath Ch 1)"})


def cheatsheet():
    return "kmchin: N = sqrt(C/(6*20)), D = 20N, checked against C = 6ND"
