# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory cell abstraction: internal state carries information through time."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_memory_cell"]

_METHOD = "Memory-cell recurrence c_t = f(c_{t-1}, x_t)"


def geron_memory_cell(c_prev, x_t, f):
    """
    Memory cell abstraction: internal state carries information through time.

    Formula: c_t = f(c_{t-1}, x_t)

    The recurrence that every recurrent architecture specialises: an
    LSTM, a GRU and a plain RNN differ only in ``f``.  ``f`` is supplied
    by the caller and the contract is enforced -- it must return a state
    of the same shape as ``c_prev``, finite, one call per time step. A
    cell that quietly changes its state's shape mid-sequence is the bug
    this check exists to catch.

    If ``x_t`` is 2-D the rows are treated as successive inputs and the
    recurrence is unrolled over them, returning the whole state
    trajectory.  The per-step change ``||c_t - c_{t-1}||`` is reported,
    since a state that stops moving is a cell that has stopped
    remembering anything new.

    Parameters
    ----------
    c_prev : array-like, shape (n_units,)
        Initial state.
    x_t : array-like, shape (n_in,) or (T, n_in)
        One input, or a sequence of ``T`` inputs.
    f : callable
        ``f(c_prev, x_t) -> c_t``, shape-preserving in ``c``.

    Returns
    -------
    result : RichResult
        Keys: c_t, states, deltas, n_steps, estimate, n, method.

    Examples
    --------
    A leaky integrator ``c <- 0.5*c + x`` run for one step from
    ``c = [2, 4]`` with ``x = [1, 1]``:

    >>> leaky = lambda c, x: 0.5 * np.asarray(c) + np.asarray(x)
    >>> r = geron_memory_cell([2.0, 4.0], [1.0, 1.0], leaky)
    >>> [float(v) for v in r["c_t"]]
    [2.0, 3.0]

    Unrolled over three identical inputs from zero the state is
    ``1, 1.5, 1.75`` -- a geometric series approaching 2:

    >>> seq = geron_memory_cell([0.0], [[1.0], [1.0], [1.0]], leaky)
    >>> [float(v[0]) for v in seq["states"]]
    [1.0, 1.5, 1.75]
    >>> seq["n_steps"]
    3

    A cell that changes the state's shape is refused:

    >>> geron_memory_cell([0.0, 0.0], [1.0, 1.0], lambda c, x: np.array([0.0]))
    Traceback (most recent call last):
        ...
    ValueError: geron_memory_cell: f returned a state of shape (1,) at step 0 but c_prev has shape (2,)

    References
    ----------
    Géron Ch 13
    """
    if not callable(f):
        raise ValueError(f"geron_memory_cell: f must be callable, got {type(f).__name__}")
    c = np.atleast_1d(np.asarray(c_prev, dtype=float)).ravel()
    if c.size == 0:
        raise ValueError("geron_memory_cell: c_prev is empty")
    if not np.all(np.isfinite(c)):
        raise ValueError("geron_memory_cell: c_prev contains non-finite values")

    X = np.asarray(x_t, dtype=float)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim != 2:
        raise ValueError(f"geron_memory_cell: x_t must be 1-D or 2-D (T, n_in), got ndim={X.ndim}")
    if X.size == 0:
        raise ValueError("geron_memory_cell: x_t is empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("geron_memory_cell: x_t contains non-finite values")

    states = []
    deltas = []
    cur = c
    for step in range(X.shape[0]):
        nxt = np.asarray(f(cur, X[step]), dtype=float).ravel()
        if nxt.shape != cur.shape:
            raise ValueError(
                f"geron_memory_cell: f returned a state of shape {nxt.shape} at step {step} "
                f"but c_prev has shape {cur.shape}"
            )
        if not np.all(np.isfinite(nxt)):
            raise ValueError(f"geron_memory_cell: f returned a non-finite state at step {step}")
        deltas.append(float(np.linalg.norm(nxt - cur)))
        cur = nxt
        states.append(cur)

    S = np.asarray(states)

    return RichResult(
        title="Memory cell",
        summary_lines=[
            ("Steps", int(S.shape[0])),
            ("Units", int(cur.size)),
            ("Final ||c||", float(np.linalg.norm(cur))),
            ("Last step change", deltas[-1]),
        ],
        interpretation=(
            "Every recurrent architecture is this recurrence with a different f; "
            "a state whose per-step change collapses to zero has stopped storing anything new."
        ),
        payload={
            "c_t": cur,
            "states": S,
            "deltas": np.asarray(deltas),
            "n_steps": int(S.shape[0]),
            "estimate": float(np.linalg.norm(cur)),
            "n": int(cur.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmcel: memory-cell recurrence c_t = f(c_{t-1}, x_t) with an enforced shape-preserving f"
