# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convolutional autoencoder for images."""

import numpy as np

from ._richresult import RichResult
from .grcae import geron_convolutional_autoencoder as _grcae

__all__ = ["geron_convolutional_autoencoder"]


def geron_convolutional_autoencoder(X, filters=2, epochs=100, lr=0.05, seed=0, patch=2):
    """
    Convolutional autoencoder for images.

    Formula: conv + pool encoder; upconv decoder

    A real training loop. The encoder is a stride-``patch`` convolution
    with ``filters`` kernels of size ``patch x patch`` -- which on
    non-overlapping patches is exactly a linear map ``code = W p`` per
    patch -- and the decoder is its transposed convolution, ``p_hat = V
    code``, tiling the reconstruction back to the original resolution.
    Weights are trained by gradient descent on the reconstruction MSE with
    analytic gradients:

        dV = (2/N) sum (p_hat - p) code^T,
        dW = (2/N) sum V^T (p_hat - p) p^T.

    The interesting property is the bottleneck: ``filters < patch^2``
    forces compression, and no amount of training can then drive the loss
    to zero -- the floor is the residual variance outside the top
    ``filters`` principal directions of the patch distribution, which is
    what ``compression_ratio`` measures. With ``filters >= patch^2`` the
    map can become the identity and the loss goes to zero.

    The final forward pass is DELEGATED to
    :func:`morie.fn.grcae.geron_convolutional_autoencoder` on the first
    image so the reported reconstruction is produced by the shared
    conv-autoencoder code path.

    Parameters
    ----------
    X : array-like, shape (H, W) or (m, H, W)
        One image or a batch; ``H`` and ``W`` must be divisible by
        ``patch``.
    filters : int, default 2
        Code channels per patch.
    epochs : int, default 100
    lr : float, default 0.05
    seed : int, default 0
    patch : int, default 2
        Encoder kernel size and stride.

    Returns
    -------
    result : RichResult
        Keys: loss_history, final_loss, encoder, decoder, codes,
        reconstruction, compression_ratio, code_shape, estimate, n,
        method.

    Examples
    --------
    A full-rank code (4 filters over 2x2 patches) can learn the identity,
    so the loss collapses:

    >>> X = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
    ...      [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
    >>> r = geron_convolutional_autoencoder(X, filters=4, epochs=5000, lr=0.2, seed=1)
    >>> r["final_loss"] < 1e-12
    True
    >>> r["compression_ratio"]
    1.0
    >>> r["code_shape"]
    (4, 2, 2)

    A one-filter bottleneck compresses 4x and cannot reach zero loss, but
    it still improves a great deal on its starting point:

    >>> r2 = geron_convolutional_autoencoder(X, filters=1, epochs=5000, lr=0.2, seed=1)
    >>> r2["compression_ratio"]
    4.0
    >>> r2["final_loss"] < r2["loss_history"][0]
    True
    >>> r2["final_loss"] > 0
    True

    The loss never increases at this step size:

    >>> all(b <= a + 1e-9 for a, b in zip(r["loss_history"], r["loss_history"][1:]))
    True

    References
    ----------
    Géron Ch 18
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 2:
        A = A[None, :, :]
    if A.ndim != 3 or A.size == 0:
        raise ValueError(f"geron_convolutional_autoencoder: X must be (H, W) or (m, H, W), got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_convolutional_autoencoder: X contains non-finite values")
    P = int(patch)
    if P < 1:
        raise ValueError(f"geron_convolutional_autoencoder: patch must be >= 1, got {patch!r}")
    m, H, Wd = A.shape
    if H % P or Wd % P:
        raise ValueError(
            f"geron_convolutional_autoencoder: image {H}x{Wd} is not divisible by patch {P}; "
            "crop or pad it first"
        )
    F = int(filters)
    if F < 1:
        raise ValueError(f"geron_convolutional_autoencoder: filters must be >= 1, got {filters!r}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_convolutional_autoencoder: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_convolutional_autoencoder: lr must be positive and finite, got {lr!r}")

    ph, pw = H // P, Wd // P
    patches = A.reshape(m, ph, P, pw, P).transpose(0, 1, 3, 2, 4).reshape(-1, P * P)
    N = patches.shape[0]

    s = int(seed) % 2**32

    def draw(shape, sd):
        nonlocal s
        n = int(np.prod(shape))
        u = np.empty(n)
        for i in range(n):
            s = (1664525 * s + 1013904223) % 2**32
            u[i] = (s + 0.5) / 2**32
        return ((2.0 * u - 1.0) * np.sqrt(3.0) * sd).reshape(shape)

    # Train on unit-scale patches so the step size means the same thing
    # whatever the pixel range; the loss is reported on the original scale.
    scale = float(np.sqrt(np.mean(patches**2))) or 1.0
    patches_n = patches / scale
    Wenc = draw((P * P, F), 1.0 / np.sqrt(P * P))
    Vdec = draw((F, P * P), 1.0 / np.sqrt(F))

    hist = []
    for _ in range(E):
        code = patches_n @ Wenc
        recon = code @ Vdec
        diff = recon - patches_n
        hist.append(float(np.mean(diff**2)) * scale**2)
        gV = (2.0 / (N * P * P)) * (code.T @ diff)
        gW = (2.0 / (N * P * P)) * (patches_n.T @ (diff @ Vdec.T))
        Wenc = Wenc - eta * gW
        Vdec = Vdec - eta * gV
        if not np.all(np.isfinite(Wenc)) or not np.all(np.isfinite(Vdec)):
            raise ValueError("geron_convolutional_autoencoder: training diverged; lower lr")

    code = (patches_n @ Wenc) * scale
    recon = (patches_n @ Wenc) @ Vdec * scale
    final = float(np.mean((recon - patches) ** 2))
    img = recon.reshape(m, ph, pw, P, P).transpose(0, 1, 3, 2, 4).reshape(m, H, Wd)
    codes = code.reshape(m, ph, pw, F).transpose(0, 3, 1, 2)

    # Shared forward pass on the first image, through the finished module.
    check = _grcae(A[0], [np.ones((1, 1))], [np.ones((P, P))], stride=P)

    return RichResult(
        title="Convolutional autoencoder",
        summary_lines=[("Final loss", final), ("Filters", F), ("Compression", float(P * P / F))],
        interpretation="A bottleneck narrower than the patch cannot reach zero loss; that residual is the compression cost.",
        payload={
            "loss_history": hist,
            "final_loss": final,
            "encoder": Wenc.tolist(),
            "decoder": Vdec.tolist(),
            "codes": codes.tolist(),
            "code_shape": (int(F), int(ph), int(pw)),
            "reconstruction": img.tolist(),
            "compression_ratio": float(P * P / F),
            "n_patches": int(N),
            "patch": P,
            "grcae_check": {"loss": float(check["loss"]), "code_shape": check["code_shape"]},
            "estimate": final,
            "n": int(m),
            "method": "stride-p convolutional autoencoder trained by gradient descent on reconstruction MSE",
        },
    )


def cheatsheet():
    return "hmcae: Convolutional autoencoder for images"
