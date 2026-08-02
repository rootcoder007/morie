# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 5.5: the RLHF objective, reward minus a KL leash."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch5_rlhf_objective"]


def _dist(p, name):
    p = np.atleast_1d(np.asarray(
        [float(v) for v in (p.values() if isinstance(p, dict) else p)],
        dtype=float))
    if p.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(p < 0):
        raise ValueError(f"{name} holds a negative probability.")
    if abs(float(p.sum()) - 1.0) > 1e-8:
        raise ValueError(
            f"{name} must sum to 1; it sums to {float(p.sum()):.6g}.")
    return p


def _kl(p, q):
    """KL(p||q) with the 0 log 0 = 0 convention; q = 0 where p > 0 is
    infinite support loss and is refused."""
    if np.any((q <= 0) & (p > 0)):
        raise ValueError(
            "pi_ref assigns zero probability where pi_theta does not; "
            "the KL divergence is infinite.")
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])))


def kamath_ch5_rlhf_objective(pi_theta, pi_ref, r_phi, beta):
    """J = max E_{y~pi_theta}[r_phi(x,y)] - beta KL(pi_theta || pi_ref).

    Both distributions run over the SAME ordered response set, and
    ``r_phi`` gives that set's rewards (a sequence, or a callable
    applied to a dict's keys). LARGER is better. beta = 0 drops the
    leash and the objective is the plain expected reward; pi_theta =
    pi_ref makes the KL exactly 0.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Eq 5.5, printed
    p. 208.

    Examples
    --------
    >>> out = kamath_ch5_rlhf_objective([0.5, 0.5], [0.5, 0.5],
    ...                                 [1.0, 0.0], 1.0)
    >>> out["estimate"], out["kl"]
    (0.5, 0.0)
    """
    p = _dist(pi_theta, "pi_theta")
    q = _dist(pi_ref, "pi_ref")
    if p.shape != q.shape:
        raise ValueError(
            f"pi_theta has {p.size} responses but pi_ref has {q.size}.")
    if callable(r_phi):
        keys = list(pi_theta.keys()) if isinstance(pi_theta, dict) else None
        if keys is None:
            raise ValueError("a callable r_phi needs named responses; pass "
                             "pi_theta as a mapping.")
        r = np.asarray([float(r_phi(k)) for k in keys], dtype=float)
    else:
        r = np.atleast_1d(np.asarray(r_phi, dtype=float))
    if r.shape != p.shape:
        raise ValueError(
            f"r_phi has {r.size} rewards for {p.size} responses.")
    beta = float(beta)
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    exp_r = float(np.sum(p * r))
    kl = _kl(p, q)
    return RichResult(payload={
        "estimate": exp_r - beta * kl, "expected_reward": exp_r,
        "kl": kl, "penalty": beta * kl, "beta": beta, "n": int(p.size),
        "method": "RLHF objective (Kamath Eq 5.5)"})


def cheatsheet():
    return "km069: E[r] - beta KL(pi_theta || pi_ref), maximise"
