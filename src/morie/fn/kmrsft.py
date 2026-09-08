# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rejection-sampling fine-tuning: keep the best-rewarded samples per
prompt and SFT on them."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_rejection_sampling_finetune"]


def kamath_rejection_sampling_finetune(prompts, samples, rewards, k,
                                       sft=None):
    """For each prompt: sample {y_i}, keep the top k by r_phi, then SFT
    on what survived.

    The selection is per PROMPT, not global -- a global top-k would
    keep every sample from the easy prompts and none from the hard
    ones, which is the failure mode this function exists to avoid.
    Ties keep the earlier sample so the retained set is reproducible.
    ``sft`` is optional: without it the retained pairs are returned
    for the caller to train on.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 5,
    rejection-sampling fine-tuning; that section is not in the 2024
    PDF, so the procedure is implemented exactly as the spec line
    states (best-of-n / RAFT).

    Examples
    --------
    >>> out = kamath_rejection_sampling_finetune(
    ...     ["p1", "p2"],
    ...     [["a", "b", "c"], ["d", "e", "f"]],
    ...     [[0.1, 0.9, 0.5], [3.0, 1.0, 2.0]], k=2)
    >>> out["retained"]
    [('p1', 'b'), ('p1', 'c'), ('p2', 'd'), ('p2', 'f')]
    >>> out["estimate"]
    4
    >>> abs(out["mean_retained_reward"] - (0.9 + 0.5 + 3.0 + 2.0) / 4) < 1e-12
    True
    """
    prompts = list(prompts)
    samples = [list(s) for s in samples]
    rewards = [list(r) for r in rewards]
    k = int(k)
    if not prompts:
        raise ValueError("no prompts supplied.")
    if len(samples) != len(prompts) or len(rewards) != len(prompts):
        raise ValueError(
            f"need one sample list and one reward list per prompt; got "
            f"{len(samples)} and {len(rewards)} for {len(prompts)} "
            "prompts.")
    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}.")
    retained, kept_rewards, dropped = [], [], 0
    for p, ys, rs in zip(prompts, samples, rewards):
        if len(ys) != len(rs):
            raise ValueError(
                f"prompt {p!r}: {len(ys)} samples but {len(rs)} rewards.")
        if not ys:
            raise ValueError(f"prompt {p!r} has no samples.")
        r = np.asarray(rs, dtype=float)
        if not np.all(np.isfinite(r)):
            raise ValueError(
                f"prompt {p!r}: a reward is non-finite, so the ranking "
                "is undefined.")
        take = min(k, len(ys))
        order = np.argsort(-r, kind="stable")[:take]
        for i in sorted(int(v) for v in order):
            retained.append((p, ys[i]))
            kept_rewards.append(float(r[i]))
        dropped += len(ys) - take
    payload = {
        "retained": retained,
        "retained_rewards": kept_rewards,
        "n_retained": len(retained), "n_dropped": dropped,
        "mean_retained_reward": float(np.mean(kept_rewards)),
        "k": k, "estimate": len(retained), "n": len(prompts),
        "method": "Rejection-sampling fine-tuning (per-prompt top-k)"}
    if sft is not None:
        if not callable(sft):
            raise ValueError("sft must be callable(retained_pairs) -> policy.")
        payload["policy"] = sft(retained)
    return RichResult(payload=payload)


def cheatsheet():
    return "kmrsft: per-prompt top-k by reward, then SFT on the survivors"
