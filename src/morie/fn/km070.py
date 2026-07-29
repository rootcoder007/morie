# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.6: the closed-form optimal KL-regularised policy."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch5_rlhf_optimal_policy"]


def kamath_ch5_rlhf_optimal_policy(pi_ref, r, beta, Z=None):
    """pi_r(y|x) = (1/Z(x)) pi_ref(y|x) exp(r(x,y) / beta).

    The exact solution of Eq 5.5: reweight the reference policy by
    exp(reward / beta) and renormalise. ``Z`` is computed when omitted;
    when supplied it is CHECKED against the sum, because a Z that does
    not normalise silently returns something that is not a
    distribution. Large beta returns pi_ref; small beta concentrates on
    the arg max reward.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.6, printed
    p. 208.

    Examples
    --------
    >>> out = kamath_ch5_rlhf_optimal_policy([0.5, 0.5], [2.0, 0.0], 2.0)
    >>> round(out["pi"][0], 10)   # e / (e + 1)
    0.7310585786
    >>> round(out["Z"], 10)
    1.8591409142
    """
    q = np.atleast_1d(np.asarray(
        [float(v) for v in (pi_ref.values() if isinstance(pi_ref, dict)
                            else pi_ref)], dtype=float))
    rv = np.atleast_1d(np.asarray(r, dtype=float))
    if q.size == 0:
        raise ValueError("pi_ref is empty.")
    if np.any(q < 0) or abs(float(q.sum()) - 1.0) > 1e-8:
        raise ValueError(
            f"pi_ref must be a distribution; it sums to {float(q.sum()):.6g}.")
    if rv.shape != q.shape:
        raise ValueError(
            f"r has {rv.size} rewards for {q.size} responses.")
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be strictly positive; exp(r/0) is "
                         "undefined.")
    w = q * np.exp(rv / beta)
    Z_hat = float(w.sum())
    if Z_hat <= 0:
        raise ValueError("the partition function is 0; pi_ref places no "
                         "mass on any response.")
    if Z is not None and abs(float(Z) - Z_hat) > 1e-8 * max(1.0, Z_hat):
        raise ValueError(
            f"the supplied Z = {float(Z):.6g} does not normalise; the sum "
            f"is {Z_hat:.6g}.")
    p = w / Z_hat
    return RichResult(payload={
        "pi": [float(v) for v in p], "Z": Z_hat, "beta": beta,
        "estimate": float(p.max()), "argmax": int(np.argmax(p)),
        "n": int(p.size),
        "method": "optimal KL-regularised policy (Kamath Eq 5.6)"})


def cheatsheet():
    return "km070: pi_r = pi_ref exp(r/beta) / Z, Z checked not assumed"
