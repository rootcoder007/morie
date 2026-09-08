# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.18: the output-projector MSE objective."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_output_projector_mse"]


def kamath_ch9_output_projector_mse(H_X, tau_X, t):
    r"""argmin over OUT_ALIGN of L_mse(H_X, tau_X(t)).

    ``tau_X`` is the modality generator's textual condition encoder
    (a callable, or a fixed target array). ``H_X`` may be a single
    projected feature matrix, or a 3-D stack of candidates, in which
    case the argmin over them is reported -- Eq 9.18's arg min made
    concrete over a finite candidate set.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.18, printed
    p. 397.

    Examples
    --------
    >>> out = kamath_ch9_output_projector_mse(
    ...     [[1.0, 2.0]], lambda tt: [[1.0, 0.0]], None)
    >>> out["estimate"]              # (0^2 + 2^2) / 2
    2.0
    """
    target = np.asarray(tau_X(t) if callable(tau_X) else tau_X,
                        dtype=float)
    H = np.asarray(H_X, dtype=float)
    if H.ndim not in (1, 2, 3):
        raise ValueError("H_X must be 1-D, a 2-D feature matrix, or a "
                         "3-D stack of candidates.")
    cands = H if H.ndim == 3 else H[None, ...]
    if cands[0].shape != target.shape:
        raise ValueError(
            f"H_X entries are {cands[0].shape} but tau_X(t) is "
            f"{target.shape}; the MSE is not defined between them.")
    if target.size == 0:
        raise ValueError("the target features are empty.")
    losses = [float(np.mean((c - target) ** 2)) for c in cands]
    k = int(np.argmin(losses))
    return RichResult(payload={
        "estimate": losses[k], "argmin": k, "losses": losses,
        "n_candidates": len(losses), "n": int(target.size),
        "method": "output-projector MSE objective (Kamath Eq 9.18)"})


def cheatsheet():
    return "km146: MSE between projected features and tau_X(t)"
