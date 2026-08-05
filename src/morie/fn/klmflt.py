# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Kalman forward recursion driven by a model specification.

Kalman (1960), Trans. ASME J. Basic Engineering 82(1):35-45,
doi:10.1115/1.3662552.  Identical recursion to the matrix-argument
form in :mod:`morie.fn.kalmf`; only the interface differs, the system
being passed as one object with entries F, H, Q, R and optionally x0
and P0 -- the shape a fitted model is usually kept in.

This module DELEGATES to kalmf rather than carrying a second copy of
the recursion.  ``morie.fn.sspace`` (Durbin & Koopman) is the
package's other entry point to the same filter; a third implementation
would be indistinguishable from correct work under any parity check,
which is exactly why there is not one here.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from .kalmf import kalman_filter as _matrix_filter

from ._richresult import RichResult

__all__ = ["kalman_filter"]


def _get(model, key, default=None):
    v = model.get(key, default) if hasattr(model, "get") else getattr(model, key, default)
    if v is None:
        if default is None:
            raise ValueError("kalman_filter: model is missing entry " + key)
        return default
    return v


def kalman_filter(y, model):
    """Filtered states and log-likelihood for a packaged state space model."""
    Y = core.mat(y)
    if len(Y) == 0:
        raise ValueError("kalman_filter: y is empty")
    F = _get(model, "F")
    H = _get(model, "H")
    Q = _get(model, "Q")
    R = _get(model, "R")
    d = len(core.mat(F))
    x0 = _get(model, "x0", [0.0] * d)
    P0 = _get(model, "P0", [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)])
    res = _matrix_filter(Y, F, H, Q, R, x0, P0)
    return RichResult(
        title="Kalman filter (model form)",
        summary_lines=[("n", len(Y)), ("state dim", d)],
        payload={
            "estimate": res["estimate"],
            "state": res["state"],
            "loglik": res["loglik"],
            "n": res["n"],
            "method": "forward predict/update recursion, Kalman (1960); delegates to kalmf",
        },
    )


def cheatsheet():
    return "klmflt: Kalman forward recursion (delegates to kalmf)"
