# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RLHF reward shaping with a KL penalty against the reference
policy."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_kl_reward_shaping"]


def kamath_kl_reward_shaping(r_phi, kl_divergence, beta):
    """r_shaped(x, y) = r_phi(x, y) - beta * KL(pi_theta || pi_ref).

    Elementwise over a batch of samples. A negative KL is refused: a
    Kullback-Leibler divergence cannot be negative, so a negative
    entry means the caller handed over a log-ratio with the wrong
    sign, and shaping with it would REWARD drifting off the reference
    policy -- exactly the failure the penalty exists to prevent.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, KL divergence
    penalty.

    Examples
    --------
    >>> out = kamath_kl_reward_shaping([1.0, 2.0], [0.5, 0.0], 0.2)
    >>> out["shaped"]
    [0.9, 2.0]
    >>> out["estimate"]
    1.45
    """
    r = np.atleast_1d(np.asarray(r_phi, dtype=float)).ravel()
    kl = np.atleast_1d(np.asarray(kl_divergence, dtype=float)).ravel()
    beta = float(beta)
    if r.size == 0:
        raise ValueError("no rewards supplied.")
    if kl.size == 1 and r.size > 1:
        kl = np.repeat(kl, r.size)
    if r.size != kl.size:
        raise ValueError(
            f"got {r.size} rewards and {kl.size} KL values.")
    if beta < 0:
        raise ValueError(
            f"beta must be non-negative; got {beta}. A negative "
            "coefficient turns the penalty into a bonus for drift.")
    if np.any(kl < 0):
        raise ValueError(
            "a KL divergence is non-negative by definition; a negative "
            "entry means the log-ratio was passed with the wrong sign.")
    shaped = r - beta * kl
    return RichResult(payload={
        "shaped": [float(v) for v in shaped],
        "estimate": float(shaped.mean()),
        "mean_reward": float(r.mean()),
        "mean_kl": float(kl.mean()),
        "penalty": float(beta * kl.mean()),
        "beta": beta, "n": int(r.size),
        "method": "KL-shaped RLHF reward r - beta * KL"})


def cheatsheet():
    return "kmklr: r_phi - beta*KL(pi||pi_ref), negative KL refused"
