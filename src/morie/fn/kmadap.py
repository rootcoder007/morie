# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 4: the Houlsby bottleneck adapter."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_houlsby_adapter"]

_erf = np.vectorize(math.erf)


def _gelu(z, approximate):
    if approximate == "none":
        return 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))
    if approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z ** 3)))
    raise ValueError("approximate must be 'none' (the exact erf GELU) "
                     "or 'tanh'.")


def kamath_houlsby_adapter(h, W_down, W_up, approximate="none"):
    r"""adapter(h) = h + W_up GELU(W_down h), with m << d.

    ``W_down`` is m x d and ``W_up`` is d x m, so the residual branch
    squeezes the d-dimensional hidden state through the m-dimensional
    bottleneck and back. ``h`` may be one hidden state (d,) or a batch
    (n, d). GELU is the exact erf form by default; ``approximate=
    'tanh'`` selects the tanh approximation.

    A bottleneck at least as wide as the model (m >= d) is refused:
    that is not the adapter of the paper, it is a full-rank layer.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Serial Adapters;
    Houlsby et al. (2019).

    Examples
    --------
    >>> import math
    >>> out = kamath_houlsby_adapter([1.0, 0.0], [[1.0, 0.0]],
    ...                              [[1.0], [0.0]])
    >>> g = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    >>> abs(out["h_adapted"][0][0] - (1.0 + g)) < 1e-12
    True
    """
    H = np.atleast_2d(np.asarray(h, dtype=float))
    Wd = np.atleast_2d(np.asarray(W_down, dtype=float))
    Wu = np.atleast_2d(np.asarray(W_up, dtype=float))
    m, d = Wd.shape
    if H.shape[1] != d:
        raise ValueError(
            f"W_down expects a {d}-dimensional hidden state; got "
            f"{H.shape[1]}.")
    if Wu.shape != (d, m):
        raise ValueError(
            f"W_up must be {d}x{m} to project back; got {Wu.shape}.")
    if m >= d:
        raise ValueError(
            f"the bottleneck m = {m} is not smaller than d = {d}; a "
            "Houlsby adapter needs m << d.")
    inner = _gelu(H @ Wd.T, approximate)
    out = H + inner @ Wu.T
    return RichResult(payload={
        "estimate": float(np.linalg.norm(out - H)),
        "h_adapted": [[float(v) for v in row] for row in out],
        "bottleneck": [[float(v) for v in row] for row in inner],
        "m": int(m), "d": int(d), "n": int(H.shape[0]),
        "method": "Houlsby bottleneck adapter (Kamath Ch 4)"})


def cheatsheet():
    return "kmadap: residual h + W_up GELU(W_down h) through m << d"
