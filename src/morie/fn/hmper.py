# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prioritized experience replay."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_prioritized_replay"]


def geron_prioritized_replay(buffer, alpha=0.6, beta=0.4, eps=1e-6, batch_size=None, seed=0):
    """
    Prioritized experience replay: sample by TD-error priority.

    Formula: p_i proportional to |delta_i|^alpha; importance sampling weights

    Sampling by |TD error| replays the transitions the network is most
    wrong about, but it changes the distribution the expectation is taken
    over, so the gradient becomes biased. The importance weights
    w_i = (N P_i)^(-beta), normalised by their maximum, undo exactly that
    much of the bias; beta is annealed to 1 by the end of training, when
    the bias matters most. alpha = 0 recovers uniform replay.

    Parameters
    ----------
    buffer : array-like
        TD errors (signed or absolute), or a sequence of dicts carrying a
        ``"td_error"`` key.
    alpha : float, default 0.6
        Prioritisation exponent in [0, 1].
    beta : float, default 0.4
        Importance-sampling exponent in [0, 1].
    eps : float, default 1e-6
        Floor added to |delta| so a zero-error transition keeps a chance.
    batch_size : int, optional
        Draw this many indices with the LCG stream.
    seed : int, default 0
        Seed for the draw.

    Returns
    -------
    result : RichResult
        Keys: priorities, probabilities, weights, indices, estimate, n,
        method.

    Examples
    --------
    With alpha = 1 the probabilities are the normalised errors:

    >>> r = geron_prioritized_replay([1.0, 1.0, 2.0], alpha=1.0, beta=1.0, eps=0.0)
    >>> [float(p) for p in r["probabilities"]]
    [0.25, 0.25, 0.5]

    Weights (N P)^-1 normalised by the maximum halve for the most-likely
    sample:

    >>> [float(w) for w in r["weights"]]
    [1.0, 1.0, 0.5]

    alpha = 0 is uniform replay:

    >>> [float(p) for p in geron_prioritized_replay([1.0, 3.0], alpha=0.0)["probabilities"]]
    [0.5, 0.5]

    References
    ----------
    Geron Ch 19
    """
    if isinstance(buffer, dict):
        raise ValueError("geron_prioritized_replay: buffer must be a sequence of transitions, not a dict")
    items = list(buffer)
    if len(items) == 0:
        raise ValueError("geron_prioritized_replay: buffer is empty")
    deltas = []
    for i, it in enumerate(items):
        if isinstance(it, dict):
            if "td_error" not in it:
                raise ValueError(f"geron_prioritized_replay: transition {i} has no 'td_error' key")
            deltas.append(float(it["td_error"]))
        else:
            deltas.append(float(np.asarray(it, dtype=float).ravel()[0]))
    d = np.abs(np.asarray(deltas, dtype=float))
    if not np.all(np.isfinite(d)):
        raise ValueError("geron_prioritized_replay: TD errors contain non-finite values")
    a, b, e = float(alpha), float(beta), float(eps)
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"geron_prioritized_replay: alpha must lie in [0, 1], got {alpha!r}")
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"geron_prioritized_replay: beta must lie in [0, 1], got {beta!r}")
    if e < 0:
        raise ValueError("geron_prioritized_replay: eps must be non-negative")

    pri = (d + e) ** a
    tot = float(pri.sum())
    if tot <= 0:
        raise ValueError("geron_prioritized_replay: every priority is zero; raise eps or alpha")
    prob = pri / tot
    N = d.size
    w = (N * prob) ** (-b)
    w = w / w.max()

    idx = None
    if batch_size is not None:
        k = int(batch_size)
        if k < 1:
            raise ValueError(f"geron_prioritized_replay: batch_size must be >= 1, got {batch_size!r}")
        s = int(seed) % 2**32
        cum = np.cumsum(prob)
        draw = np.empty(k, dtype=int)
        for i in range(k):
            s = (1664525 * s + 1013904223) % 2**32
            u = (s + 0.5) / 2**32
            draw[i] = min(int(np.searchsorted(cum, u)), N - 1)
        idx = draw

    return RichResult(
        title="Prioritized experience replay",
        summary_lines=[("Transitions", int(N)), ("alpha", a), ("beta", b)],
        interpretation="Priority sampling biases the gradient; the IS weights remove as much of it as beta asks for.",
        payload={
            "priorities": pri,
            "probabilities": prob,
            "weights": w,
            "indices": idx,
            "alpha": a,
            "beta": b,
            "estimate": prob,
            "n": int(N),
            "method": "Proportional prioritized replay with importance-sampling weights",
        },
    )


def cheatsheet():
    return "hmper: Prioritized experience replay probabilities and IS weights"
