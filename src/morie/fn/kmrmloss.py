# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reward-model training loss over preference pairs (Bradley-Terry
negative log-likelihood)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_reward_model_training_loss"]


def kamath_reward_model_training_loss(scores_w, scores_l):
    """L_RM = -E[log sigmoid(r_phi(x, y_w) - r_phi(x, y_l))].

    Only the DIFFERENCE of the two scores enters, which is why a
    reward model is identified up to an additive constant -- adding
    1000 to every score leaves this loss unchanged. Computed as
    ``log(1 + exp(-d))`` via ``logaddexp``, so a large positive margin
    underflows to 0 instead of overflowing exp.

    The pairwise accuracy (how often r_w > r_l) is reported next to
    the loss: a loss near log 2 with 50% accuracy is a model that has
    learned nothing, and the loss alone will not say so.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, reward model
    training.

    Examples
    --------
    >>> import math
    >>> out = kamath_reward_model_training_loss([1.0, 2.0], [1.0, 2.0])
    >>> abs(out["estimate"] - math.log(2)) < 1e-12
    True
    >>> out["accuracy"]
    0.0
    >>> good = kamath_reward_model_training_loss([5.0], [0.0])
    >>> abs(good["estimate"] - math.log(1 + math.exp(-5.0))) < 1e-12
    True
    >>> good["accuracy"]
    1.0
    """
    w = np.atleast_1d(np.asarray(scores_w, dtype=float)).ravel()
    l = np.atleast_1d(np.asarray(scores_l, dtype=float)).ravel()
    if w.size != l.size:
        raise ValueError(
            f"{w.size} chosen scores against {l.size} rejected ones; "
            "the loss is over PAIRS.")
    if w.size == 0:
        raise ValueError("no preference pairs supplied.")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(l))):
        raise ValueError("reward scores must be finite.")
    d = w - l
    per = np.logaddexp(0.0, -d)
    loss = float(per.mean())
    return RichResult(payload={
        "estimate": loss, "loss": loss,
        "per_pair": [float(v) for v in per],
        "margins": [float(v) for v in d],
        "mean_margin": float(d.mean()),
        "accuracy": float(np.mean(d > 0)),
        "n": int(w.size),
        "method": "Bradley-Terry reward-model NLL over preference pairs"})


def cheatsheet():
    return "kmrmloss: mean log(1+exp(-(r_w - r_l))); accuracy reported too"
