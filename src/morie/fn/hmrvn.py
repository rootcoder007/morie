# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RevNet: reversible residual blocks enabling activation-free backprop."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_revnet"]


def geron_revnet(x, F, G):
    """
    RevNet: reversible residual blocks enabling activation-free backprop.

    Formula: y1 = x1 + F(x2); y2 = x2 + G(y1); reversible

    The point of a reversible block is that the inputs can be recovered
    from the outputs, so activations need not be stored for the backward
    pass. That claim is *verified* here rather than asserted: the block is
    inverted (``x2 = y2 - G(y1)``, ``x1 = y1 - F(x2)``) and the maximum
    reconstruction error is returned. The inversion holds for arbitrary F
    and G, which is why they may be any caller-supplied callables.

    Parameters
    ----------
    x : array-like
        Input with an even number of columns (last axis), split in half.
    F, G : callable
        Residual functions. Each is called with an array shaped like one
        half and must return an array of that same shape. Anything else
        is an error, not a broadcast.

    Returns
    -------
    result : RichResult
        Keys: y, y1, y2, x_reconstructed, reconstruction_error,
        estimate, n, method.

    Examples
    --------
    x1 = [1, 2], x2 = [3, 4] with F(a) = 2a and G(a) = a + 1 gives
    y1 = [1, 2] + [6, 8] = [7, 10] and y2 = [3, 4] + [8, 11] = [11, 15]:

    >>> r = geron_revnet([1.0, 2.0, 3.0, 4.0], lambda a: 2 * a, lambda a: a + 1)
    >>> [float(v) for v in r["y"]]
    [7.0, 10.0, 11.0, 15.0]
    >>> float(r["reconstruction_error"])
    0.0
    >>> [float(v) for v in r["x_reconstructed"]]
    [1.0, 2.0, 3.0, 4.0]

    References
    ----------
    Géron Ch 12
    """
    X = np.asarray(x, dtype=float)
    if X.size == 0:
        raise ValueError("geron_revnet: x is empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("geron_revnet: x contains non-finite values")
    if X.shape[-1] % 2:
        raise ValueError(
            f"geron_revnet: the split axis has {X.shape[-1]} entries; a reversible block needs an even width"
        )
    if not callable(F) or not callable(G):
        raise ValueError("geron_revnet: F and G must both be callables mapping one half to a same-shaped array")

    half = X.shape[-1] // 2
    x1 = X[..., :half]
    x2 = X[..., half:]

    def _apply(fn, name, arg):
        out = np.asarray(fn(arg), dtype=float)
        if out.shape != arg.shape:
            raise ValueError(
                f"geron_revnet: {name} returned shape {out.shape} but must return {arg.shape}; "
                "a reversible block needs shape-preserving residual functions"
            )
        if not np.all(np.isfinite(out)):
            raise ValueError(f"geron_revnet: {name} returned non-finite values")
        return out

    y1 = x1 + _apply(F, "F", x2)
    y2 = x2 + _apply(G, "G", y1)

    # Inversion -- the whole reason the block exists.
    x2_rec = y2 - _apply(G, "G", y1)
    x1_rec = y1 - _apply(F, "F", x2_rec)
    x_rec = np.concatenate([x1_rec, x2_rec], axis=-1)
    err = float(np.max(np.abs(x_rec - X)))

    y = np.concatenate([y1, y2], axis=-1)

    return RichResult(
        title="Reversible residual block (RevNet)",
        summary_lines=[("Half width", half), ("Reconstruction error", err)],
        interpretation=(
            "Because the block inverts exactly, backprop can recompute activations from the outputs; "
            "memory becomes O(1) in depth at the cost of one extra forward pass per block."
        ),
        payload={
            "y": y,
            "y1": y1,
            "y2": y2,
            "x1": x1,
            "x2": x2,
            "x_reconstructed": x_rec,
            "reconstruction_error": err,
            "reversible": bool(err <= 1e-9 * max(1.0, float(np.max(np.abs(X))))),
            "estimate": err,
            "n": int(X.shape[-1]),
            "method": "RevNet block y1 = x1 + F(x2), y2 = x2 + G(y1), verified by explicit inversion",
        },
    )


def cheatsheet():
    return "hmrvn: RevNet: reversible residual blocks enabling activation-free backprop"


# compact alias per ledger/NAMING.md
geronrevnet = geron_revnet
