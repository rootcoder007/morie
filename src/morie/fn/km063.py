# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.10: the VeRA forward pass."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch4_vera_forward"]


def _diag(v, name, size):
    v = np.atleast_1d(np.asarray(v, dtype=float))
    if v.ndim == 2:
        if v.shape[0] != v.shape[1] or not np.allclose(v, np.diag(np.diag(v))):
            raise ValueError(f"{name} must be diagonal.")
        v = np.diag(v)
    if v.shape[0] != size:
        raise ValueError(
            f"{name} has {v.shape[0]} entries but needs {size}.")
    return v


def kamath_ch4_vera_forward(W_0, Lambda_b, Lambda_d, A, B, x):
    """h = W_0 x + Lambda_b B Lambda_d A x.

    A and B are FROZEN random matrices shared across layers; only the
    two diagonal vectors Lambda_b (length d) and Lambda_d (length r)
    are trained, which is where VeRA's parameter count of d + r per
    layer comes from -- reported as ``n_trainable`` against LoRA's
    r(d + k).

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.10, printed
    p. 155.

    Examples
    --------
    >>> out = kamath_ch4_vera_forward([[1.0, 0.0], [0.0, 1.0]],
    ...     [3.0, 1.0], [2.0], [[1.0, 0.0]], [[1.0], [0.0]], [1.0, 2.0])
    >>> out["h"]
    [7.0, 2.0]
    >>> out["n_trainable"], out["n_trainable_lora"]
    (3, 4)
    """
    W0 = np.atleast_2d(np.asarray(W_0, dtype=float))
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    Bm = np.atleast_2d(np.asarray(B, dtype=float))
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    if xv.ndim != 1:
        raise ValueError("x must be a vector.")
    d, k = W0.shape
    if k != xv.shape[0]:
        raise ValueError(
            f"W_0 has {k} columns but x has {xv.shape[0]} entries.")
    if Am.shape[1] != k:
        raise ValueError(f"A has {Am.shape[1]} columns but needs {k}.")
    if Bm.shape[1] != Am.shape[0]:
        raise ValueError(
            f"B is {Bm.shape} and A is {Am.shape}; r must match.")
    if Bm.shape[0] != d:
        raise ValueError(f"B has {Bm.shape[0]} rows but needs {d}.")
    r = Am.shape[0]
    lb = _diag(Lambda_b, "Lambda_b", d)
    ld = _diag(Lambda_d, "Lambda_d", r)
    base = W0 @ xv
    delta = lb * (Bm @ (ld * (Am @ xv)))
    return RichResult(payload={
        "h": [float(v) for v in base + delta],
        "base": [float(v) for v in base],
        "delta_h": [float(v) for v in delta],
        "r": int(r), "n_trainable": int(d + r),
        "n_trainable_lora": int(r * (d + k)),
        "estimate": float((base + delta)[0]), "n": int(xv.shape[0]),
        "method": "VeRA forward pass (Kamath Eq 4.10)"})


def cheatsheet():
    return "km063: h = W_0 x + Lam_b B Lam_d A x, only diagonals train"
