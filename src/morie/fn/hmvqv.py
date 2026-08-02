# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Discrete VAE (VQ-VAE): vector-quantized latents with codebook."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_vq_vae", "quantize"]


def _lcg(shape, seed, scale=0.5):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def quantize(z_e, codebook):
    """Nearest-codebook-entry assignment: ``k_i = argmin_k ||z_e,i - e_k||``.

    Returns ``(indices, z_q)``.
    """
    d2 = np.sum((z_e[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)
    return idx, codebook[idx]


def geron_vq_vae(X, codebook_size=4, latent_dim=2, epochs=200, lr=0.05, beta=0.25, seed=0):
    """
    Discrete VAE (VQ-VAE): vector-quantized latents with codebook.

    Formula: z = codebook[argmin_k ||z_e - e_k||]; commitment + codebook losses

    The three-term objective is implemented exactly as in van den Oord et
    al. (2017):

    ``L = ||x - decode(z_q)||^2 + ||sg[z_e] - e||^2 + beta*||z_e - sg[e]||^2``

    where ``sg`` is the stop-gradient. The first term trains the decoder
    and -- via the **straight-through estimator**, which copies the
    decoder gradient at ``z_q`` straight onto ``z_e`` -- the encoder too,
    because argmin has zero gradient everywhere and would otherwise cut
    the encoder off entirely. The second term moves the codebook towards
    the encodings; the third keeps the encoder from running away from the
    codebook.

    Parameters
    ----------
    X : array-like
        Training data (n, d).
    codebook_size : int, default 4
        Number of discrete codes K (>= 2, <= n).
    latent_dim : int, default 2
        Width of each code vector (>= 1).
    epochs : int, default 200
        Gradient steps (>= 1).
    lr : float, default 0.05
        Learning rate (> 0).
    beta : float, default 0.25
        Commitment weight (>= 0).
    seed : int, default 0
        LCG seed for the encoder, decoder and codebook.

    Returns
    -------
    result : RichResult
        Keys: codes, indices, z_e, z_q, codebook, reconstruction,
        recon_error, codebook_loss, commitment_loss, perplexity,
        loss_curve, estimate, n, method.

    Examples
    --------
    Two well-separated groups and two codes: the quantiser assigns one
    code per group and every latent equals its codebook entry exactly.

    >>> import numpy as np
    >>> X = [[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]]
    >>> r = geron_vq_vae(X, codebook_size=2, latent_dim=1, epochs=400, lr=0.05)
    >>> int(len(set(int(i) for i in r["indices"])))
    2
    >>> bool(np.allclose(r["z_q"], r["codebook"][r["indices"]]))
    True
    >>> bool(r["loss_curve"][-1] < r["loss_curve"][0])
    True
    >>> bool(1.0 <= r["perplexity"] <= 2.0)
    True

    References
    ----------
    Géron Ch 18
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_vq_vae: X must be a non-empty (n, d) matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_vq_vae: X contains non-finite values")
    n, d = A.shape
    K = int(codebook_size)
    if K < 2:
        raise ValueError(f"geron_vq_vae: codebook_size must be >= 2, got {K}")
    if K > n:
        raise ValueError(f"geron_vq_vae: codebook_size {K} exceeds the {n} training points")
    k = int(latent_dim)
    if k < 1:
        raise ValueError(f"geron_vq_vae: latent_dim must be >= 1, got {k}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_vq_vae: epochs must be >= 1, got {E}")
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"geron_vq_vae: lr must be positive and finite, got {step}")
    b = float(beta)
    if not np.isfinite(b) or b < 0:
        raise ValueError(f"geron_vq_vae: beta must be non-negative and finite, got {b}")

    We = _lcg((d, k), int(seed) + 1)
    be = np.zeros(k)
    Wd = _lcg((k, d), int(seed) + 2)
    bd = np.zeros(d)
    # Codebook initialised on k-means++-style spread encodings of the data.
    z0 = A @ We + be
    cb = np.vstack([z0[int(i * (n - 1) / max(1, K - 1))] for i in range(K)]) + _lcg((K, k), int(seed) + 3, 0.01)

    losses = []
    for _ in range(E):
        z_e = A @ We + be
        idx, z_q = quantize(z_e, cb)
        xhat = z_q @ Wd + bd
        diff = xhat - A
        recon = float(np.mean(diff * diff))
        cb_loss = float(np.mean((z_q - z_e) ** 2))  # ||sg[z_e] - e||^2
        commit = float(np.mean((z_e - z_q) ** 2))  # ||z_e - sg[e]||^2
        losses.append(recon + cb_loss + b * commit)

        dxhat = 2.0 * diff / (n * d)
        dWd = z_q.T @ dxhat
        dbd = dxhat.sum(axis=0)
        dz_q = dxhat @ Wd.T
        # straight-through: the decoder gradient lands on z_e unchanged
        dz_e = dz_q + b * 2.0 * (z_e - z_q) / (n * k)
        dWe = A.T @ dz_e
        dbe = dz_e.sum(axis=0)
        dcb = np.zeros_like(cb)
        for j in range(K):
            m = idx == j
            if np.any(m):
                dcb[j] = 2.0 * np.sum(cb[j] - z_e[m], axis=0) / (n * k)

        We = We - step * dWe
        be = be - step * dbe
        Wd = Wd - step * dWd
        bd = bd - step * dbd
        cb = cb - step * dcb

    z_e = A @ We + be
    idx, z_q = quantize(z_e, cb)
    xhat = z_q @ Wd + bd
    recon = float(np.mean((xhat - A) ** 2))
    cb_loss = float(np.mean((z_q - z_e) ** 2))
    counts = np.bincount(idx, minlength=K).astype(float)
    p = counts / counts.sum()
    nz = p > 0
    perplexity = float(np.exp(-np.sum(p[nz] * np.log(p[nz]))))

    return RichResult(
        title="VQ-VAE",
        summary_lines=[
            ("Codebook size", K),
            ("Codes used", int(np.sum(counts > 0))),
            ("Reconstruction MSE", recon),
            ("Codebook perplexity", perplexity),
        ],
        interpretation=(
            "Quantising the latent kills the gradient, so the straight-through estimator is not an "
            "optimisation -- without it the encoder receives nothing. Low perplexity means codebook collapse."
        ),
        payload={
            "codes": z_q,
            "indices": idx,
            "z_e": z_e,
            "z_q": z_q,
            "codebook": cb,
            "counts": counts,
            "reconstruction": xhat,
            "recon_error": recon,
            "codebook_loss": cb_loss,
            "commitment_loss": cb_loss,
            "perplexity": perplexity,
            "loss_curve": np.asarray(losses, dtype=float),
            "beta": b,
            "estimate": recon,
            "n": int(n),
            "method": "VQ-VAE: nearest-code quantisation, straight-through encoder gradient, codebook + commitment losses",
        },
    )


def cheatsheet():
    return "hmvqv: Discrete VAE (VQ-VAE): vector-quantized latents with codebook"
