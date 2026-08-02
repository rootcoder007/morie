# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hebb's rule: connections strengthen when pre- and post-synaptic activities co-occur."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_hebb_rule"]

_METHOD = "Hebbian weight update"


def geron_hebb_rule(X, Y, eta=0.1, W=None):
    """
    Hebb's rule: connections strengthen when pre- and post-synaptic activities co-occur.

    Formula: dw_ij = eta * x_i * y_j

    Accumulated over the supplied batch this is ``dW = eta * X^T Y``, an
    outer-product update with no error signal anywhere in it -- which is
    also why plain Hebbian learning is unstable: nothing bounds the
    weights, so ``||W||`` grows without limit under repeated exposure.
    The growth is reported so that instability is visible rather than
    surprising.

    Parameters
    ----------
    X : array-like, shape (m, n_pre) or (n_pre,)
        Pre-synaptic activities; one row per presentation.
    Y : array-like, shape (m, n_post) or (n_post,)
        Post-synaptic activities, same number of rows as ``X``.
    eta : float
        Learning rate (positive).
    W : array-like, shape (n_pre, n_post), optional
        Existing weights to update; default zeros.

    Returns
    -------
    result : RichResult
        Keys: dW, W, norm_before, norm_after, estimate, n, method.

    Examples
    --------
    One presentation, one outer product: ``dW = 0.5 * [1,2]^T [3]``.

    >>> r = geron_hebb_rule([[1.0, 2.0]], [[3.0]], eta=0.5)
    >>> [float(v) for v in r["dW"].ravel()]
    [1.5, 3.0]

    Two presentations accumulate: ``[1,0]^T[1] + [0,1]^T[1] = I``
    scaled by eta:

    >>> r2 = geron_hebb_rule([[1.0, 0.0], [0.0, 1.0]], [[1.0], [1.0]], eta=1.0)
    >>> [float(v) for v in r2["dW"].ravel()]
    [1.0, 1.0]

    Anti-correlated activity weakens the connection:

    >>> float(geron_hebb_rule([[1.0]], [[-2.0]], eta=1.0)["dW"][0, 0])
    -2.0

    References
    ----------
    Géron Ch 9
    """
    A = np.asarray(X, dtype=float)
    B = np.asarray(Y, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"geron_hebb_rule: X and Y must be 1-D or 2-D, got ndim {A.ndim} and {B.ndim}")
    if A.size == 0 or B.size == 0:
        raise ValueError("geron_hebb_rule: X and Y must be non-empty")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"geron_hebb_rule: X has {A.shape[0]} presentations but Y has {B.shape[0]}"
        )
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError("geron_hebb_rule: X and Y must be finite")
    lr = float(eta)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"geron_hebb_rule: eta must be a positive finite learning rate, got {eta!r}")

    n_pre, n_post = A.shape[1], B.shape[1]
    W0 = np.zeros((n_pre, n_post)) if W is None else np.asarray(W, dtype=float)
    if W0.shape != (n_pre, n_post):
        raise ValueError(f"geron_hebb_rule: W has shape {W0.shape}, expected {(n_pre, n_post)}")

    dW = lr * (A.T @ B)
    W_new = W0 + dW
    nb = float(np.linalg.norm(W0))
    na = float(np.linalg.norm(W_new))

    return RichResult(
        title="Hebbian update",
        summary_lines=[("Connections", n_pre * n_post), ("||W|| before", nb), ("||W|| after", na)],
        warnings=(
            ["Hebbian learning has no stabilising term: repeated exposure grows ||W|| without bound."]
            if na > nb
            else []
        ),
        interpretation="Cells that fire together wire together; the update is a pure outer product.",
        payload={
            "dW": dW,
            "W": W_new,
            "norm_before": nb,
            "norm_after": na,
            "estimate": float(np.linalg.norm(dW)),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhebb: Hebb's rule dW = eta * X^T Y (outer product, no error signal)"
