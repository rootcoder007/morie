# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 5: best-of-N (rejection) sampling."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_best_of_n_sampling"]


def kamath_best_of_n_sampling(samples, rewards=None, reward_fn=None,
                              x=None):
    r"""y_hat = argmax_{y in {y_1..y_N}} r_phi(x, y).

    Either pass the reward-model scores in ``rewards``, or a callable
    ``reward_fn(x, y)`` that is applied to every sample. Ties go to
    the first sample, so the choice is deterministic; the reward
    spread is reported because best-of-N is only worth its N when the
    rewards actually differ.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, Best-of-N Sampling.

    Examples
    --------
    >>> out = kamath_best_of_n_sampling(["a", "b", "c"],
    ...                                 [0.1, 0.9, 0.4])
    >>> out["best"], out["best_index"], out["estimate"]
    ('b', 1, 0.9)
    """
    ys = list(samples)
    if len(ys) == 0:
        raise ValueError("no samples were generated; best-of-0 has no "
                         "argmax.")
    if rewards is None:
        if not callable(reward_fn):
            raise ValueError("give rewards= or a callable reward_fn(x, "
                             "y).")
        r = np.array([float(reward_fn(x, y)) for y in ys])
    else:
        r = np.atleast_1d(np.asarray(rewards, dtype=float))
        if r.size != len(ys):
            raise ValueError(
                f"{r.size} rewards for {len(ys)} samples.")
    if not np.all(np.isfinite(r)):
        raise ValueError("the reward model returned non-finite scores.")
    k = int(np.argmax(r))
    return RichResult(payload={
        "estimate": float(r[k]), "best": ys[k], "best_index": k,
        "rewards": [float(v) for v in r],
        "reward_spread": float(r.max() - r.min()), "n": len(ys),
        "method": "best-of-N sampling (Kamath Ch 5)"})


def cheatsheet():
    return "kmbon: pick the highest reward-model score among N samples"
