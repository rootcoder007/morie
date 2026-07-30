# morie.fn -- shared helpers (rootcoder007/morie)
"""Shared state plumbing for the first-order optimiser update rules.

Every update rule in this family is one step of an iterative method, so it
needs somewhere to keep its accumulators between calls.  Rather than make
each caller hand-roll that, the rules take an optional ``state`` mapping and
return the updated one, so a training loop is::

    st = None
    for _ in range(n_steps):
        r = adam(grad(theta), state=st)
        theta = theta + r["update"]
        st = r["state"]

Passing ``state=None`` starts a fresh run.  The returned ``state`` is a plain
dict of arrays and is safe to pickle or checkpoint.
"""

from __future__ import annotations

import numpy as np

__all__ = ["as_vector", "init_state", "step_result"]


def as_vector(g, name="g"):
    """Coerce a gradient to a 1-D float array, rejecting empty/non-finite input."""
    a = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def init_state(state, n, keys=("m", "v")):
    """Return a working copy of ``state``, creating zeroed accumulators.

    ``t`` is the 1-based step counter the bias corrections need.
    """
    out = {"t": 0}
    if state:
        out.update({k: np.asarray(state[k], dtype=float).copy() for k in keys if k in state})
        out["t"] = int(state.get("t", 0))
    for k in keys:
        if k not in out or np.shape(out[k]) != (n,):
            out[k] = np.zeros(n)
    out["t"] += 1
    return out


def step_result(update, state, method, **extra):
    """Assemble the common payload for one optimiser step."""
    from ._richresult import RichResult

    update = np.asarray(update, dtype=float)
    payload = {
        "update": update,
        "state": state,
        "step_norm": float(np.linalg.norm(update)),
        "t": int(state["t"]),
        "method": method,
    }
    payload.update(extra)
    return RichResult(
        title=method,
        summary_lines=[("step", int(state["t"])), ("|update|", float(np.linalg.norm(update)))],
        payload=payload,
    )
