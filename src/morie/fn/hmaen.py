# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Undercomplete linear autoencoder (PCA equivalent when linear)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_autoencoder"]


def geron_autoencoder(X, bottleneck, center=True):
    """
    Undercomplete linear autoencoder (PCA equivalent when linear).

    Formula: min ||x - W_2 W_1 x||^2 with bottleneck dim < input dim

    For a linear autoencoder the global optimum of the reconstruction loss
    is spanned by the top principal components, so the encoder is solved in
    closed form by an SVD rather than by gradient descent: W_1 is the matrix
    of leading right singular vectors and W_2 = W_1^T.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Training data.
    bottleneck : int
        Code width; must satisfy 1 <= bottleneck <= d.
    center : bool, default True
        Subtract the feature means before encoding (and add them back on
        decode), which is what makes the solution equal to PCA.

    Returns
    -------
    result : RichResult
        Keys: encoder, decoder, codes, reconstruction, recon_error,
        explained_variance_ratio, estimate, n, method.

    Examples
    --------
    Points on a line are reconstructed exactly from one code unit:

    >>> r = geron_autoencoder([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], 1)
    >>> round(float(r["recon_error"]), 12)
    0.0
    >>> round(float(r["explained_variance_ratio"][0]), 12)
    1.0
    >>> r["encoder"].shape, r["codes"].shape
    ((1, 2), (4, 1))

    A full-width bottleneck is the identity, so the error is zero there too:

    >>> round(float(geron_autoencoder([[1.0, 5.0], [2.0, 1.0]], 2)["recon_error"]), 12)
    0.0

    References
    ----------
    Géron Ch 18
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_autoencoder: X must be 2-D, got ndim={A.ndim}")
    n, d = A.shape
    if n == 0:
        raise ValueError("geron_autoencoder: X has no rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_autoencoder: X must be finite")
    k = int(bottleneck)
    if k < 1 or k > d:
        raise ValueError(f"geron_autoencoder: bottleneck must lie in [1, {d}], got {k}")

    mean = A.mean(axis=0) if center else np.zeros(d)
    Xc = A - mean
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    W1 = Vt[:k, :]
    W2 = W1.T
    codes = Xc @ W2
    recon = codes @ W1 + mean
    err = float(np.mean(np.sum((A - recon) ** 2, axis=1)))

    total = float(np.sum(sv**2))
    evr = (sv**2 / total) if total > 0 else np.zeros_like(sv)

    def encode(Xnew, _W2=W2, _mean=mean, _d=d):
        B = np.asarray(Xnew, dtype=float)
        if B.ndim == 1:
            B = B.reshape(1, -1)
        if B.shape[1] != _d:
            raise ValueError(f"encode: expected {_d} features, got {B.shape[1]}")
        return (B - _mean) @ _W2

    def decode(C, _W1=W1, _mean=mean, _k=k):
        Cm = np.asarray(C, dtype=float)
        if Cm.ndim == 1:
            Cm = Cm.reshape(1, -1)
        if Cm.shape[1] != _k:
            raise ValueError(f"decode: expected {_k} code units, got {Cm.shape[1]}")
        return Cm @ _W1 + _mean

    return RichResult(
        title="Linear undercomplete autoencoder",
        summary_lines=[
            ("Bottleneck", k),
            ("Reconstruction MSE (per sample)", err),
            ("Variance retained", float(np.sum(evr[:k]))),
        ],
        payload={
            "encoder": W1,
            "decoder": W2,
            "codes": codes,
            "reconstruction": recon,
            "recon_error": err,
            "explained_variance_ratio": evr,
            "mean": mean,
            "encode": encode,
            "decode": decode,
            "estimate": err,
            "n": int(n),
            "method": "Linear undercomplete autoencoder solved in closed form by SVD (equals PCA)",
        },
    )


def cheatsheet():
    return "hmaen: Undercomplete linear autoencoder (PCA equivalent when linear)"
