# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Law of large numbers: running mean convergence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_lln"]


def wasserman_lln(data):
    """
    Weak law of large numbers, made observable.

    Formula: X_bar_n -> mu in probability. The payload carries the
    full running-mean path X_bar_1, X_bar_2, ..., X_bar_n and the
    final mean, so convergence can be inspected rather than asserted.

    Parameters
    ----------
    data : array-like
        Sample (non-empty), in observation order.

    Returns
    -------
    result : dict
        Keys: estimate (final running mean), running_means,
        last_gap (|X_bar_n - X_bar_{n-1}|, nan for n = 1), n, method.

    References
    ----------
    Wasserman (2004), Ch 5, Theorem 5.6.

    Examples
    --------
    >>> out = wasserman_lln([2.0, 4.0, 6.0])
    >>> out["running_means"]
    [2.0, 3.0, 4.0]
    >>> out["estimate"]
    4.0
    >>> out["last_gap"]
    1.0
    >>> wasserman_lln([])
    Traceback (most recent call last):
        ...
    ValueError: the running mean of an empty sample is undefined.
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = data.size
    if n == 0:
        raise ValueError("the running mean of an empty sample is undefined.")
    running = np.cumsum(data) / np.arange(1, n + 1)
    gap = float(abs(running[-1] - running[-2])) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": float(running[-1]),
        "running_means": [float(v) for v in running],
        "last_gap": gap, "n": int(n),
        "method": "LLN running means X_bar_1..X_bar_n"})


def cheatsheet():
    return "wsmlln: running means cumsum/1..n; final mean is the estimate"


# compact alias per ledger/NAMING.md
wassermanlln = wasserman_lln
