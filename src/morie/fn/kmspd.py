# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Speculative decoding: accept or reject a draft token, then sample
the residual."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_speculative_decoding"]


def _check_dist(p, name):
    p = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
    if p.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(p < 0):
        raise ValueError(f"{name} has a negative probability.")
    s = p.sum()
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(
            f"{name} sums to {s}, not 1; it is not a distribution.")
    return p


def kamath_speculative_decoding(draft_probs, target_probs, proposed=None,
                                u=None):
    """accept(t) = min(1, p_target(t) / p_draft(t)); on rejection sample
    from the normalised residual max(0, p_target - p_draft).

    That residual is the entire correctness argument: the accept step
    plus the residual step together sample EXACTLY from the target
    distribution, so speculative decoding is lossless rather than an
    approximation. It is returned so a caller can check that, and its
    total mass equals 1 - sum_t min(p_draft, p_target), the expected
    rejection rate.

    ``proposed`` defaults to the draft's argmax. ``u`` is the uniform
    draw for the accept test; supply it for a deterministic decision
    (no RNG is created here).

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 8,
    speculative decoding; that section is not in the 2024 PDF, so the
    rule is implemented exactly as the spec line states (Leviathan
    et al. 2023).

    Examples
    --------
    >>> out = kamath_speculative_decoding([0.5, 0.5], [0.25, 0.75],
    ...                                   proposed=0, u=0.4)
    >>> out["estimate"]
    0.5
    >>> out["accepted"]
    True
    >>> out["residual"]
    [0.0, 1.0]
    >>> rej = kamath_speculative_decoding([0.5, 0.5], [0.25, 0.75],
    ...                                   proposed=0, u=0.9)
    >>> rej["accepted"], rej["resample_from_residual"]
    (False, True)
    >>> abs(rej["rejection_rate"] - 0.25) < 1e-12
    True
    """
    pd = _check_dist(draft_probs, "draft_probs")
    pt = _check_dist(target_probs, "target_probs")
    if pd.size != pt.size:
        raise ValueError(
            f"the draft covers {pd.size} tokens and the target "
            f"{pt.size}; they must share a vocabulary.")
    t = int(np.argmax(pd)) if proposed is None else int(proposed)
    if not 0 <= t < pd.size:
        raise ValueError(f"the proposed token must lie in [0, {pd.size - 1}].")
    if pd[t] == 0:
        raise ValueError(
            f"the draft model assigns probability 0 to token {t}, so it "
            "cannot have proposed it and the acceptance ratio is 0/0.")
    ratio = float(pt[t] / pd[t])
    accept_p = min(1.0, ratio)
    resid = np.maximum(0.0, pt - pd)
    mass = float(resid.sum())
    resid_norm = (resid / mass) if mass > 0 else np.zeros_like(resid)
    payload = {
        "estimate": accept_p, "accept_prob": accept_p, "ratio": ratio,
        "proposed": t,
        "residual": [float(v) for v in resid_norm],
        "residual_mass": mass,
        "rejection_rate": float(1.0 - np.minimum(pd, pt).sum()),
        "n": int(pd.size),
        "method": "Speculative decoding accept/reject with residual "
                  "resampling"}
    if u is not None:
        u = float(u)
        if not 0.0 <= u <= 1.0:
            raise ValueError(f"u must lie in [0, 1]; got {u}.")
        acc = u < accept_p
        payload["accepted"] = bool(acc)
        payload["resample_from_residual"] = bool(not acc)
        if not acc and mass == 0:
            raise ValueError(
                "the token was rejected but the residual has zero mass; "
                "with p_target <= p_draft everywhere no rejection can "
                "occur, so the inputs are inconsistent.")
    return RichResult(payload=payload)


def cheatsheet():
    return "kmspd: min(1, p_t/p_d) accept; residual max(0, p_t-p_d) on reject"
