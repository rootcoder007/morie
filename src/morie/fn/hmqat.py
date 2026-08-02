# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantization-aware training with a straight-through estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_quantization_aware_training"]


def _fake_quant(w, bits):
    """Symmetric fake quantization: round to the grid, return float."""
    qmax = 2 ** (bits - 1) - 1
    scale = float(np.max(np.abs(w))) / qmax
    if scale == 0.0:
        return w.copy(), 0.0
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale, scale


def geron_quantization_aware_training(model, X, y, epochs=200, lr=0.1, bits=8):
    """
    Quantization-aware training (QAT): simulate quantization during training.

    Formula: fake quant nodes in forward; straight-through estimator backward

    The forward pass uses weights snapped to the quantization grid, so
    the loss the optimiser sees is the loss the deployed integer model
    will have. The backward pass pretends the rounding was the identity
    -- the straight-through estimator -- because the true derivative of
    round() is zero almost everywhere and would kill every gradient.

    The model here is a linear regressor, the smallest thing that shows
    the mechanism honestly: the loss is measured at the QUANTIZED
    weights, the gradient is taken with respect to the latent
    full-precision ones, and both are returned so the gap is visible.

    Parameters
    ----------
    model : array-like
        Initial full-precision weights (a bias column in ``X`` is yours
        to add).
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    epochs : int, default 200
    lr : float, default 0.1
        Learning rate (positive).
    bits : int, default 8
        Bit width, 2 to 16.

    Returns
    -------
    result : RichResult
        Keys: weights, quantized_weights, scale, loss, loss_history,
        fp_loss, estimate, n, method.

    Examples
    --------
    y = 2x is recovered to within a quantization step:

    >>> r = geron_quantization_aware_training([0.0], [[1.0], [2.0], [3.0]],
    ...                                       [2.0, 4.0, 6.0], epochs=300, lr=0.05)
    >>> bool(abs(float(r["quantized_weights"][0]) - 2.0) < 0.05)
    True
    >>> bool(r["loss"] < 1e-3)
    True

    The loss never rises over training on this convex problem:

    >>> h = r["loss_history"]
    >>> bool(h[-1] <= h[0])
    True

    Two bits leave one positive level, so the fit is visibly coarser:

    >>> c = geron_quantization_aware_training([0.0], [[1.0], [2.0]], [2.0, 4.0],
    ...                                       epochs=300, lr=0.05, bits=2)
    >>> bool(c["loss"] >= r["loss"])
    True

    References
    ----------
    Geron Appendix B
    """
    w = np.atleast_1d(np.asarray(model, dtype=float)).astype(float).copy()
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    yv = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_quantization_aware_training: X must be a non-empty 2-D array, got shape {A.shape}")
    if yv.size != A.shape[0]:
        raise ValueError(
            f"geron_quantization_aware_training: X has {A.shape[0]} rows but y has {yv.size} entries"
        )
    if w.size != A.shape[1]:
        raise ValueError(
            f"geron_quantization_aware_training: model has {w.size} weights but X has {A.shape[1]} columns"
        )
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_quantization_aware_training: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("geron_quantization_aware_training: lr must be positive and finite")
    b = int(bits)
    if not (2 <= b <= 16):
        raise ValueError(f"geron_quantization_aware_training: bits must lie in [2, 16], got {bits!r}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yv)) or not np.all(np.isfinite(w)):
        raise ValueError("geron_quantization_aware_training: inputs contain non-finite values")

    m = A.shape[0]
    hist = []
    scale = 0.0
    for _ in range(E):
        wq, scale = _fake_quant(w, b)
        resid = A @ wq - yv
        loss = float(np.mean(resid**2))
        hist.append(loss)
        # Straight-through: dL/dw = dL/dwq, the rounding is treated as identity.
        grad = (2.0 / m) * (A.T @ resid)
        w = w - eta * grad
    wq, scale = _fake_quant(w, b)
    resid = A @ wq - yv
    loss = float(np.mean(resid**2))
    hist.append(loss)
    fp_resid = A @ w - yv
    fp_loss = float(np.mean(fp_resid**2))

    return RichResult(
        title="Quantization-aware training",
        summary_lines=[("Bits", b), ("Quantized loss", loss), ("Full-precision loss", fp_loss)],
        interpretation="The STE pretends round() is the identity; without it every gradient would be zero.",
        payload={
            "weights": w,
            "quantized_weights": wq,
            "scale": scale,
            "loss": loss,
            "fp_loss": fp_loss,
            "loss_history": hist,
            "bits": b,
            "estimate": wq,
            "n": int(m),
            "method": "QAT on a linear model: fake-quant forward, straight-through backward",
        },
    )


def cheatsheet():
    return "hmqat: Quantization-aware training with a straight-through estimator"
