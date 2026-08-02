# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Policy evaluation: pi(a|s) or a deterministic pi(s)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_policy"]


def geron_policy(state, pi, seed=0):
    """
    Policy pi(a|s) or deterministic pi(s).

    Formula: a_t ~ pi(. | s_t)

    A policy is just the map from state to behaviour; the interesting
    question is how much randomness it keeps. The entropy of pi(.|s) is
    returned because it is the exploration budget: at zero entropy the
    agent is greedy and can never discover a better action, which is why
    Geron's policy-gradient agents keep a stochastic policy rather than
    an argmax one.

    ``pi`` may be a callable ``pi(state) -> probabilities``, a table with
    one row per state, a dict keyed by state, or an integer/array of
    integers for a deterministic policy.

    Parameters
    ----------
    state : int or hashable
        Current state.
    pi : callable, array-like or dict
        The policy.
    seed : int, default 0
        Seed of the integer LCG used for the sampled action, so the draw
        is reproducible on every machine.

    Returns
    -------
    result : RichResult
        Keys: probabilities, action, greedy_action, entropy,
        deterministic, estimate, n, method.

    Examples
    --------
    >>> r = geron_policy(0, [[0.25, 0.75]])
    >>> [float(p) for p in r["probabilities"]]
    [0.25, 0.75]
    >>> int(r["greedy_action"]), round(float(r["entropy"]), 6)
    (1, 0.562335)

    A deterministic policy has zero entropy:

    >>> d = geron_policy(1, {0: 0, 1: 2})
    >>> int(d["action"]), float(d["entropy"]), bool(d["deterministic"])
    (2, 0.0, True)

    References
    ----------
    Geron Ch 19
    """
    if callable(pi):
        raw = pi(state)
    elif isinstance(pi, dict):
        if state not in pi:
            raise ValueError(f"geron_policy: state {state!r} is not in the policy table")
        raw = pi[state]
    else:
        table = np.asarray(pi)
        if table.ndim == 1:
            si = int(state)
            if not (0 <= si < table.size):
                raise ValueError(f"geron_policy: state {si} outside the {table.size} entries of pi")
            raw = table[si]
        elif table.ndim == 2:
            si = int(state)
            if not (0 <= si < table.shape[0]):
                raise ValueError(f"geron_policy: state {si} outside the {table.shape[0]} rows of pi")
            raw = table[si]
        else:
            raise ValueError(f"geron_policy: pi must be 1-D or 2-D, got ndim={table.ndim}")

    arr = np.atleast_1d(np.asarray(raw, dtype=float))
    if arr.size == 1 and float(arr[0]) == int(arr[0]) and np.ndim(raw) == 0:
        action = int(arr[0])
        if action < 0:
            raise ValueError(f"geron_policy: deterministic action must be a non-negative index, got {action}")
        probs = np.zeros(action + 1)
        probs[action] = 1.0
        deterministic = True
    else:
        probs = arr.ravel()
        if probs.size == 0:
            raise ValueError("geron_policy: the policy returned no action probabilities")
        if np.any(probs < 0) or not np.all(np.isfinite(probs)):
            raise ValueError("geron_policy: action probabilities must be finite and non-negative")
        total = float(probs.sum())
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"geron_policy: action probabilities sum to {total}, not 1")
        deterministic = bool(np.max(probs) == 1.0)
        action = None

    nz = probs[probs > 0]
    entropy = float(-np.sum(nz * np.log(nz))) + 0.0
    greedy = int(np.argmax(probs))
    if action is None:
        s = int(seed) % 2**32
        s = (1664525 * s + 1013904223) % 2**32
        u = (s + 0.5) / 2**32
        action = int(np.searchsorted(np.cumsum(probs), u * probs.sum()))
        action = min(action, probs.size - 1)
    return RichResult(
        title="Policy",
        summary_lines=[("Actions", int(probs.size)), ("Entropy (nats)", entropy), ("Greedy action", greedy)],
        interpretation="Entropy is the exploration budget; a greedy policy cannot discover a better action.",
        payload={
            "probabilities": probs,
            "action": int(action),
            "greedy_action": greedy,
            "entropy": entropy,
            "deterministic": deterministic,
            "estimate": int(action),
            "n": int(probs.size),
            "method": "Policy evaluation with entropy and a reproducible sampled action",
        },
    )


def cheatsheet():
    return "hmpol: Policy pi(a|s) with entropy and sampled action"
