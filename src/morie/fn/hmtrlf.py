# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fine-tuning via TRL (Transformer Reinforcement Learning) library."""

import numpy as np

from ._richresult import RichResult
from .hmsft import geron_sft

__all__ = ["geron_trl_finetune"]


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def geron_trl_finetune(
    model=None, dataset=None, method="sft", epochs=200, lr=0.1, beta=0.1, clip_eps=0.2, theta_ref=None
):
    """
    Fine-tuning via TRL (Transformer Reinforcement Learning) library.

    Formula: SFT, DPO, PPO trainers from TRL

    The three trainers, with their actual objectives:

    * ``"sft"`` -- delegated to :func:`morie.fn.hmsft.geron_sft`; the
      dataset is a list of ``(instruction, response)`` pairs.
    * ``"dpo"`` -- Direct Preference Optimisation. Dataset items are
      ``(x_chosen, x_rejected)`` feature vectors and the loss is
      ``-log sigmoid(beta * (Delta_policy - Delta_ref))`` with
      ``Delta = logp(chosen) - logp(rejected)``. No reward model and no
      sampling: the preference *is* the training signal, and the reference
      term is what stops the policy drifting arbitrarily far from where it
      started.
    * ``"ppo"`` -- the clipped surrogate. Items are
      ``(x, logp_old, advantage)`` and the objective is
      ``min(r*A, clip(r, 1-eps, 1+eps)*A)`` with ``r = exp(logp - logp_old)``.
      The clip is not a safety rail on the gradient magnitude: once the
      ratio leaves the trust region *in the improving direction* the
      gradient is exactly zero, which is what stops one batch destroying
      the policy.

    Parameters
    ----------
    model : array-like, optional
        Initial parameters; zeros by default.
    dataset : sequence
        Shape depends on `method` (see above). Required, non-empty.
    method : {"sft", "dpo", "ppo"}, default "sft"
        Trainer.
    epochs : int, default 200
        Gradient steps (>= 1).
    lr : float, default 0.1
        Learning rate (> 0).
    beta : float, default 0.1
        DPO temperature (> 0).
    clip_eps : float, default 0.2
        PPO clip range (> 0).
    theta_ref : array-like, optional
        Reference policy for DPO; zeros by default.

    Returns
    -------
    result : RichResult
        Keys: theta, loss, loss_curve, margin (DPO), clipped_fraction (PPO),
        estimate, n, method.

    Examples
    --------
    DPO on one preference pair: the loss falls and the policy's margin
    between chosen and rejected grows.

    >>> pairs = [([1.0, 0.0], [0.0, 1.0]), ([1.0, 1.0], [0.0, 1.0])]
    >>> r = geron_trl_finetune(None, pairs, method="dpo", epochs=300, lr=0.5, beta=1.0)
    >>> bool(r["loss_curve"][-1] < r["loss_curve"][0])
    True
    >>> bool(r["margin"] > 0)
    True
    >>> round(float(r["loss_curve"][0]), 9) == round(float(np.log(2)), 9)
    True

    PPO with a positive advantage: once the ratio passes 1 + eps the
    surrogate is clipped and that sample stops contributing gradient.

    >>> items = [([1.0], 0.0, 1.0)]
    >>> r2 = geron_trl_finetune(None, items, method="ppo", epochs=50, lr=0.5, clip_eps=0.2)
    >>> float(r2["clipped_fraction"])
    1.0
    >>> bool(r2["ratio"][0] >= 1.2)
    True
    >>> round(float(r2["loss"]), 12)
    -1.2

    References
    ----------
    Géron Ch 15
    """
    if dataset is None:
        raise ValueError("geron_trl_finetune: dataset is required")
    m = str(method).lower()
    if m not in ("sft", "dpo", "ppo"):
        raise ValueError(f"geron_trl_finetune: method must be 'sft', 'dpo' or 'ppo', got {method!r}")
    data = list(dataset)
    if not data:
        raise ValueError("geron_trl_finetune: dataset is empty")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_trl_finetune: epochs must be >= 1, got {E}")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_trl_finetune: lr must be positive and finite, got {step}")

    if m == "sft":
        inner = geron_sft(model, data, epochs=E, lr=step)
        return RichResult(
            title="TRL SFT trainer",
            summary_lines=[("Demonstrations", int(inner["n"])), ("Mean NLL", float(inner["loss"]))],
            interpretation="SFT is maximum likelihood on demonstrations; see hmsft for the head that was trained.",
            payload={
                "theta": np.asarray(inner["W"], dtype=float),
                "loss": float(inner["loss"]),
                "loss_curve": np.asarray(inner["loss_curve"], dtype=float),
                "accuracy": float(inner["accuracy"]),
                "trainer": "sft",
                "estimate": float(inner["loss"]),
                "n": int(inner["n"]),
                "method": "TRL SFT trainer (delegated to hmsft)",
            },
        )

    if m == "dpo":
        b = float(beta)
        if not np.isfinite(b) or b <= 0:
            raise ValueError(f"geron_trl_finetune: beta must be positive and finite, got {b}")
        chosen, rejected = [], []
        for i, item in enumerate(data):
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise ValueError(f"geron_trl_finetune: dpo item {i} must be (x_chosen, x_rejected)")
            chosen.append(np.asarray(item[0], dtype=float).ravel())
            rejected.append(np.asarray(item[1], dtype=float).ravel())
        C = np.vstack(chosen)
        R = np.vstack(rejected)
        if C.shape != R.shape:
            raise ValueError("geron_trl_finetune: chosen and rejected features must have the same shape")
        if not (np.all(np.isfinite(C)) and np.all(np.isfinite(R))):
            raise ValueError("geron_trl_finetune: dpo features must be finite")
        d = C.shape[1]
        theta = np.zeros(d) if model is None else np.asarray(model, dtype=float).ravel()
        if theta.size != d:
            raise ValueError(f"geron_trl_finetune: model has {theta.size} parameters but the features have {d}")
        ref = np.zeros(d) if theta_ref is None else np.asarray(theta_ref, dtype=float).ravel()
        if ref.size != d:
            raise ValueError(f"geron_trl_finetune: theta_ref has {ref.size} parameters but the features have {d}")
        D = C - R
        ref_delta = D @ ref
        losses = []
        for _ in range(E):
            z = b * (D @ theta - ref_delta)
            losses.append(float(np.mean(-np.log(np.maximum(_sigmoid(z), np.finfo(float).tiny)))))
            g = -(_sigmoid(-z) * b)[:, None] * D
            theta = theta - step * g.mean(axis=0)
        z = b * (D @ theta - ref_delta)
        losses.append(float(np.mean(-np.log(np.maximum(_sigmoid(z), np.finfo(float).tiny)))))
        margin = float(np.mean(D @ theta))
        return RichResult(
            title="TRL DPO trainer",
            summary_lines=[("Preference pairs", int(C.shape[0])), ("beta", b), ("Loss", losses[-1]), ("Margin", margin)],
            interpretation=(
                "DPO skips the reward model entirely: the Bradley-Terry preference likelihood is "
                "optimised directly, with the reference policy acting as the KL anchor."
            ),
            payload={
                "theta": theta,
                "loss": losses[-1],
                "loss_curve": np.asarray(losses, dtype=float),
                "margin": margin,
                "beta": b,
                "trainer": "dpo",
                "estimate": losses[-1],
                "n": int(C.shape[0]),
                "method": "DPO: -log sigmoid(beta * (policy margin - reference margin)) by gradient descent",
            },
        )

    eps = float(clip_eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"geron_trl_finetune: clip_eps must be positive and finite, got {eps}")
    feats, logp_old, adv = [], [], []
    for i, item in enumerate(data):
        if not (isinstance(item, (tuple, list)) and len(item) == 3):
            raise ValueError(f"geron_trl_finetune: ppo item {i} must be (x, logp_old, advantage)")
        feats.append(np.asarray(item[0], dtype=float).ravel())
        logp_old.append(float(item[1]))
        adv.append(float(item[2]))
    Xp = np.vstack(feats)
    lo = np.asarray(logp_old)
    A = np.asarray(adv)
    if not (np.all(np.isfinite(Xp)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(A))):
        raise ValueError("geron_trl_finetune: ppo inputs must be finite")
    d = Xp.shape[1]
    theta = np.zeros(d) if model is None else np.asarray(model, dtype=float).ravel()
    if theta.size != d:
        raise ValueError(f"geron_trl_finetune: model has {theta.size} parameters but the features have {d}")

    losses = []
    clipped = 0.0
    for _ in range(E):
        ratio = np.exp(np.clip(Xp @ theta - lo, -30, 30))
        unclipped = ratio * A
        clip_r = np.clip(ratio, 1 - eps, 1 + eps)
        obj = np.minimum(unclipped, clip_r * A)
        losses.append(float(-np.mean(obj)))
        active = unclipped <= clip_r * A  # gradient flows only through the unclipped branch
        clipped = float(np.mean(~active))
        g = np.where(active, ratio * A, 0.0)[:, None] * Xp
        theta = theta + step * g.mean(axis=0)
    ratio = np.exp(np.clip(Xp @ theta - lo, -30, 30))
    obj = np.minimum(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)
    losses.append(float(-np.mean(obj)))

    return RichResult(
        title="TRL PPO trainer",
        summary_lines=[
            ("Samples", int(Xp.shape[0])),
            ("clip eps", eps),
            ("Surrogate loss", losses[-1]),
            ("Clipped fraction", clipped),
        ],
        interpretation=(
            "The clipped objective is flat outside the trust region, so a batch that would move the "
            "policy too far simply stops producing gradient rather than being scaled down."
        ),
        payload={
            "theta": theta,
            "loss": losses[-1],
            "loss_curve": np.asarray(losses, dtype=float),
            "ratio": ratio,
            "clipped_fraction": clipped,
            "clip_eps": eps,
            "trainer": "ppo",
            "estimate": losses[-1],
            "n": int(Xp.shape[0]),
            "method": "PPO clipped surrogate min(rA, clip(r, 1-eps, 1+eps)A) by gradient ascent",
        },
    )


def cheatsheet():
    return "hmtrlf: Fine-tuning via TRL (Transformer Reinforcement Learning) library"
