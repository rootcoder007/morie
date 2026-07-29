# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Denoising autoencoder: reconstruct clean input from corrupted."""

import numpy as np

from ._richresult import RichResult
from .grdae import geron_denoising_autoencoder as _grdae
from .hmdfw import lcg_normal

__all__ = ["geron_denoising_autoencoder"]


def geron_denoising_autoencoder(X, noise_std=0.3, epochs=300, lr=0.05, hidden=None, seed=0):
    """
    Denoising autoencoder: reconstruct clean input from corrupted.

    Formula: min ||x - decode(encode(x + noise))||^2

    A real training loop. Gaussian noise of scale ``noise_std`` is added
    to the inputs (deterministic LCG/Box-Muller, so runs reproduce), the
    corrupted batch is passed through a linear encoder/decoder pair, and
    the loss is measured against the *clean* target -- which is the whole
    idea: the target is not the input the network saw, so the identity map
    is no longer the trivial solution.

    Gradients are analytic: ``dV = (2/N) E^T (E V - X)`` and
    ``dW = (2/N) Xt^T ((E V - X) V^T)`` with ``Xt`` the corrupted input
    and ``E = Xt W``.

    The final evaluation is DELEGATED to
    :func:`morie.fn.grdae.geron_denoising_autoencoder`, which computes the
    loss, the noise energy and the denoising gain from the clean input,
    the noise and the reconstruction.

    ``passthrough_loss`` is the baseline to beat: the error you would get
    by returning the corrupted input unchanged. A ``final_loss`` below it
    means the network actually removed noise rather than memorising it.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Clean training data.
    noise_std : float, default 0.3
        Standard deviation of the additive corruption; must be positive
        (a denoising autoencoder with no noise is just an autoencoder).
    epochs : int, default 300
    lr : float, default 0.05
    hidden : int, optional
        Code width; default ``d``.
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: loss_history, final_loss, encoder, decoder, reconstruction,
        clean_loss, denoising_gain, snr_db, noise_energy,
        passthrough_loss, estimate, n, method.

    Examples
    --------
    Data on a line in 2-D, corrupted off it: the autoencoder learns to
    project back onto the line, so the reconstruction error is well below
    the noise energy.

    >>> X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    >>> r = geron_denoising_autoencoder(X, noise_std=0.2, epochs=800, lr=0.02, hidden=1, seed=1)
    >>> r["final_loss"] < r["passthrough_loss"]
    True
    >>> round(r["passthrough_loss"], 6)
    0.050201
    >>> len(r["loss_history"])
    800

    Training reduces the loss:

    >>> r["loss_history"][-1] < r["loss_history"][0]
    True

    Zero noise is rejected, since it makes the problem a different one:

    >>> geron_denoising_autoencoder(X, noise_std=0.0)
    Traceback (most recent call last):
      ...
    ValueError: geron_denoising_autoencoder: noise_std must be positive; with no corruption this is a plain autoencoder (see hmaen)

    References
    ----------
    Géron Ch 18
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_denoising_autoencoder: X must be a non-empty (m, d) array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_denoising_autoencoder: X contains non-finite values")
    sd = float(noise_std)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(
            "geron_denoising_autoencoder: noise_std must be positive; with no corruption this is a plain autoencoder (see hmaen)"
        )
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_denoising_autoencoder: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_denoising_autoencoder: lr must be positive and finite, got {lr!r}")
    m, d = A.shape
    h = d if hidden is None else int(hidden)
    if h < 1:
        raise ValueError(f"geron_denoising_autoencoder: hidden must be >= 1, got {hidden!r}")

    noise = sd * lcg_normal((m, d), seed + 1)
    Xt = A + noise

    s = int(seed) % 2**32

    def draw(shape, scale):
        nonlocal s
        n = int(np.prod(shape))
        u = np.empty(n)
        for i in range(n):
            s = (1664525 * s + 1013904223) % 2**32
            u[i] = (s + 0.5) / 2**32
        return ((2.0 * u - 1.0) * np.sqrt(3.0) * scale).reshape(shape)

    Wenc = draw((d, h), 1.0 / np.sqrt(d))
    Vdec = draw((h, d), 1.0 / np.sqrt(h))

    hist = []
    for _ in range(E):
        code = Xt @ Wenc
        rec = code @ Vdec
        diff = rec - A
        hist.append(float(np.mean(diff**2)))
        gV = (2.0 / (m * d)) * (code.T @ diff)
        gW = (2.0 / (m * d)) * (Xt.T @ (diff @ Vdec.T))
        Wenc = Wenc - eta * gW
        Vdec = Vdec - eta * gV
        if not np.all(np.isfinite(Wenc)) or not np.all(np.isfinite(Vdec)):
            raise ValueError("geron_denoising_autoencoder: training diverged; lower lr")

    rec = (Xt @ Wenc) @ Vdec
    final = float(np.mean((rec - A) ** 2))
    ev = _grdae(A, noise, rec, corruption="additive")

    return RichResult(
        title="Denoising autoencoder",
        summary_lines=[("Final loss", final), ("Noise sd", sd), ("Denoising gain", float(ev["denoising_gain"]))],
        interpretation="The target is the clean input, so the identity map is no longer a solution.",
        payload={
            "loss_history": hist,
            "final_loss": final,
            "encoder": Wenc.tolist(),
            "decoder": Vdec.tolist(),
            "reconstruction": rec.tolist(),
            "corrupted": Xt.tolist(),
            "noise": noise.tolist(),
            "clean_loss": float(ev["loss"]),
            "denoising_gain": float(ev["denoising_gain"]),
            "snr_db": float(ev["snr_db"]),
            "noise_energy": float(ev["noise_energy"]),
            "passthrough_loss": float(np.mean(noise**2)),
            "hidden": int(h),
            "noise_std": sd,
            "estimate": final,
            "n": int(m),
            "method": "linear denoising autoencoder trained on corrupted inputs; evaluation delegated to grdae",
        },
    )


def cheatsheet():
    return "hmdae: Denoising autoencoder: reconstruct clean input from corrupted"
