# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hidden layers guideline: add layers until validation error stops improving."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_hidden_layers_heuristic"]

_METHOD = "Depth selection by validation-error plateau"


def geron_hidden_layers_heuristic(model, X, y, max_layers=10, min_layers=1, patience=2, tol=1e-4, val_fraction=0.2, seed=0):
    """
    Hidden layers guideline: add layers until validation error stops improving.

    Formula: depth L chosen so val_err(L) converges

    Depth is searched with early stopping rather than exhaustively: keep
    adding layers while the validation error keeps falling by more than
    ``tol``, and stop after ``patience`` consecutive layers that fail to
    improve on the best seen so far.  The selected depth is the one with
    the lowest validation error, not the last one tried -- those differ
    exactly when the search overshoots, which is the whole point of
    keeping ``patience`` above 1.

    ``model`` is caller-supplied and its contract is enforced:
    ``model(n_layers, X_train, y_train, X_val, y_val) -> float``, a
    finite validation error (lower is better).  Anything else raises with
    the depth named -- a scorer that silently returns NaN would make
    every comparison False and select depth ``min_layers`` by accident.

    Parameters
    ----------
    model : callable
        ``model(n_layers, X_train, y_train, X_val, y_val) -> val_error``.
    X : array-like, shape (m, n)
        Features.
    y : array-like, shape (m,)
        Targets.
    max_layers, min_layers : int
        Depth range to search, ``1 <= min_layers <= max_layers``.
    patience : int
        Consecutive non-improving depths tolerated before stopping.
    tol : float
        Minimum improvement that counts as an improvement.
    val_fraction : float
        Fraction held out for validation.
    seed : int
        Seed for the split.

    Returns
    -------
    result : RichResult
        Keys: best_n_layers, best_error, errors, depths_tried,
        stopped_early, estimate, n, method.

    Examples
    --------
    An error curve ``1/(1+L)`` keeps improving, so the search runs to
    the cap and picks the deepest:

    >>> X = [[float(i)] for i in range(20)]
    >>> y = [float(i) for i in range(20)]
    >>> decreasing = lambda L, Xt, yt, Xv, yv: 1.0 / (1.0 + L)
    >>> r = geron_hidden_layers_heuristic(decreasing, X, y, max_layers=5)
    >>> r["best_n_layers"], round(r["best_error"], 6)
    (5, 0.166667)
    >>> r["stopped_early"]
    False

    A curve that bottoms out at 3 layers and then gets worse stops two
    layers later and still reports 3:

    >>> vshape = lambda L, Xt, yt, Xv, yv: abs(L - 3) + 0.5
    >>> v = geron_hidden_layers_heuristic(vshape, X, y, max_layers=10, patience=2)
    >>> v["best_n_layers"], v["depths_tried"], v["stopped_early"]
    (3, 5, True)

    A model that returns something non-numeric is refused, with the
    depth named:

    >>> geron_hidden_layers_heuristic(lambda *a: float("nan"), X, y, max_layers=2)
    Traceback (most recent call last):
        ...
    ValueError: geron_hidden_layers_heuristic: model returned a non-finite validation error (nan) at depth 1

    References
    ----------
    Géron Ch 9
    """
    if not callable(model):
        raise ValueError(f"geron_hidden_layers_heuristic: model must be callable, got {type(model).__name__}")
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yy = np.asarray(y, dtype=float).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_hidden_layers_heuristic: X must be a non-empty 2-D array, got shape {A.shape}")
    if A.shape[0] != yy.size:
        raise ValueError(f"geron_hidden_layers_heuristic: X has {A.shape[0]} rows but y has {yy.size} entries")
    lo, hi = int(min_layers), int(max_layers)
    if lo < 1 or hi < lo:
        raise ValueError(
            f"geron_hidden_layers_heuristic: need 1 <= min_layers <= max_layers, got {min_layers!r}, {max_layers!r}"
        )
    pat = int(patience)
    if pat < 1:
        raise ValueError(f"geron_hidden_layers_heuristic: patience must be at least 1, got {patience!r}")
    t = float(tol)
    if not np.isfinite(t) or t < 0:
        raise ValueError(f"geron_hidden_layers_heuristic: tol must be finite and non-negative, got {tol!r}")
    vf = float(val_fraction)
    if not (0.0 < vf < 1.0):
        raise ValueError(f"geron_hidden_layers_heuristic: val_fraction must lie in (0, 1), got {val_fraction!r}")

    m = A.shape[0]
    n_val = int(round(m * vf))
    if n_val < 1 or m - n_val < 1:
        raise ValueError(
            f"geron_hidden_layers_heuristic: {m} rows with val_fraction={vf} leaves {n_val} validation rows"
        )
    perm = np.random.default_rng(int(seed)).permutation(m)
    vi, ti = perm[:n_val], perm[n_val:]
    Xt, yt, Xv, yv = A[ti], yy[ti], A[vi], yy[vi]

    errors = {}
    best_L = None
    best_err = np.inf
    stale = 0
    stopped = False
    for L in range(lo, hi + 1):
        err = model(L, Xt, yt, Xv, yv)
        try:
            err = float(err)
        except (TypeError, ValueError):
            raise ValueError(
                f"geron_hidden_layers_heuristic: model returned {type(err).__name__} at depth {L}, expected a float"
            ) from None
        if not np.isfinite(err):
            raise ValueError(
                f"geron_hidden_layers_heuristic: model returned a non-finite validation error ({err}) at depth {L}"
            )
        errors[L] = err
        if err < best_err - t:
            best_err = err
            best_L = L
            stale = 0
        else:
            stale += 1
            if stale >= pat:
                stopped = True
                break

    if best_L is None:
        best_L = lo
        best_err = errors[lo]

    return RichResult(
        title="Depth selection",
        summary_lines=[
            ("Best depth", best_L),
            ("Validation error", best_err),
            ("Depths tried", len(errors)),
            ("Stopped early", stopped),
        ],
        tables=[{"title": "Validation error by depth", "headers": ["layers", "val_error"],
                 "rows": [[L, e] for L, e in errors.items()]}],
        interpretation=(
            "The reported depth is the best seen, not the last tried; patience above 1 is what lets "
            "the search step past a flat spot without settling in it."
        ),
        payload={
            "best_n_layers": int(best_L),
            "best_error": float(best_err),
            "errors": errors,
            "depths_tried": len(errors),
            "stopped_early": stopped,
            "estimate": float(best_err),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhplm: choose depth by validation-error plateau with patience; reports the best depth, not the last"
