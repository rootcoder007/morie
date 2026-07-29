# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel PCA with sigmoid kernel."""

import numpy as np

from ._richresult import RichResult
from .hmkprbf import kernel_pca_from_gram

__all__ = ["geron_kernel_pca_sigmoid"]

_METHOD = "Kernel PCA (sigmoid / tanh kernel)"


def geron_kernel_pca_sigmoid(X, n_components, gamma=None, coef0=1.0):
    """
    Kernel PCA with sigmoid kernel.

    Formula: K(x,y) = tanh(gamma * x^T y + coef0)

    The sigmoid kernel is only *conditionally* positive semi-definite:
    for many ``(gamma, coef0)`` pairs the Gram matrix has genuinely
    negative eigenvalues, so it is not an inner product in any Hilbert
    space and "kernel PCA" is being run on something that is not a
    kernel.  Rather than clip that away silently, the number and size of
    the negative eigenvalues are reported and warned about; components
    are taken from the positive end of the spectrum only.

    The centring and eigen-step are delegated to
    :func:`morie.fn.hmkprbf.kernel_pca_from_gram`.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Data.
    n_components : int
        Components to keep.
    gamma : float, optional
        Scale on the inner product; defaults to ``1/n_features``.
    coef0 : float
        Offset inside the tanh.

    Returns
    -------
    result : RichResult
        Keys: X_projected, eigenvalues, alphas, K, n_negative_eigenvalues,
        is_psd, estimate, n, method.

    Examples
    --------
    The kernel entry is exactly ``tanh(gamma x.y + coef0)``; for
    ``x = 1, y = 2, gamma = 1, coef0 = 0`` that is ``tanh(2)``:

    >>> r = geron_kernel_pca_sigmoid([[1.0], [2.0], [3.0]], n_components=1,
    ...                              gamma=1.0, coef0=0.0)
    >>> round(float(r["K"][0, 1]), 9)
    0.96402758

    The kernel is bounded by 1 in absolute value, whatever the data:

    >>> bool(np.all(np.abs(r["K"]) <= 1.0))
    True

    Non-PSD-ness is reported rather than hidden:

    >>> r["is_psd"] in (True, False)
    True
    >>> r["X_projected"].shape
    (3, 1)

    References
    ----------
    Géron Ch 7
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_kernel_pca_sigmoid: X must be a non-empty 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_kernel_pca_sigmoid: X contains non-finite values")
    m, n_feat = A.shape
    g = 1.0 / n_feat if gamma is None else float(gamma)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"geron_kernel_pca_sigmoid: gamma must be positive and finite, got {gamma!r}")
    c0 = float(coef0)
    if not np.isfinite(c0):
        raise ValueError(f"geron_kernel_pca_sigmoid: coef0 must be finite, got {coef0!r}")

    K = np.tanh(g * (A @ A.T) + c0)
    proj, vals, alphas, Kc = kernel_pca_from_gram(K, n_components)

    spectrum = np.linalg.eigvalsh(0.5 * (Kc + Kc.T))
    tol = 1e-8 * max(1.0, float(np.max(np.abs(spectrum))))
    n_neg = int(np.count_nonzero(spectrum < -tol))
    is_psd = n_neg == 0

    warns = []
    if not is_psd:
        warns.append(
            f"the centred sigmoid kernel has {n_neg} negative eigenvalue(s) "
            f"(most negative {float(spectrum.min()):.4g}): it is not a valid inner-product kernel at "
            f"gamma={g}, coef0={c0}, so the embedding is heuristic."
        )

    return RichResult(
        title="Kernel PCA (sigmoid)",
        summary_lines=[
            ("Samples", int(m)),
            ("gamma / coef0", f"{g} / {c0}"),
            ("Negative eigenvalues", n_neg),
        ],
        warnings=warns,
        interpretation="tanh saturates, so every kernel entry lies in (-1, 1) regardless of the data scale.",
        payload={
            "X_projected": proj,
            "eigenvalues": vals,
            "alphas": alphas,
            "K": K,
            "K_centered": Kc,
            "spectrum": spectrum,
            "n_negative_eigenvalues": n_neg,
            "is_psd": is_psd,
            "gamma": g,
            "coef0": c0,
            "estimate": float(vals[0]),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkpsg: sigmoid kernel PCA tanh(gamma x.y + coef0), flagging its non-PSD spectrum"
