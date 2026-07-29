# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Denoising autoencoder: reconstruct clean x from corrupted x_tilde."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_denoising_autoencoder"]

_METHOD = "Denoising autoencoder reconstruction loss"


def geron_denoising_autoencoder(x, noise, decoded, corruption="additive"):
    r"""Reconstruction loss measured against the **clean** input.

    .. math::
        \tilde x = x + \text{noise}, \qquad
        L = \bigl\|x - \text{Dec}(\text{Enc}(\tilde x))\bigr\|^2

    The point of the corruption is that the loss is scored against
    :math:`x`, not :math:`\tilde x` -- so copying the input verbatim is
    now a *losing* strategy, and the code has to encode structure that
    survives the noise.  Comparing ``loss`` to ``noise_energy`` says
    whether the model actually denoised: a loss above the noise energy
    means the network did worse than passing the corrupted input through
    untouched.

    Parameters
    ----------
    x : array-like, shape (m, d)
        Clean inputs.
    noise : array-like
        Corruption, broadcastable to ``x``. With
        ``corruption="dropout"`` this is a keep-mask of 0/1 instead.
    decoded : array-like, shape (m, d)
        Reconstructions.
    corruption : {"additive", "dropout"}, optional
        How ``noise`` combines with ``x``.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``mse_per_element``, ``x_tilde``,
        ``noise_energy``, ``denoising_gain``, ``snr_db``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 18, Denoising Autoencoders section.

    Examples
    --------
    A perfect denoiser recovers ``x`` exactly, so the loss is zero even
    though the input was corrupted:

    >>> r = geron_denoising_autoencoder([[1.0, 2.0]], [[0.1, -0.1]], [[1.0, 2.0]])
    >>> r["loss"]
    0.0
    >>> [round(v, 6) for v in r["x_tilde"][0]]
    [1.1, 1.9]
    >>> round(r["noise_energy"], 6)
    0.02

    Getting one coordinate wrong by 1 costs 1:

    >>> r2 = geron_denoising_autoencoder([[1.0, 2.0]], [[0.1, -0.1]], [[1.0, 3.0]])
    >>> round(r2["loss"], 6)
    1.0
    >>> r2["denoising_gain"] < 1.0
    True
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    decoded = np.atleast_2d(np.asarray(decoded, dtype=float))
    noise = np.asarray(noise, dtype=float)
    if x.size == 0:
        raise ValueError("x is empty.")
    if x.shape != decoded.shape:
        raise ValueError(f"decoded shape {decoded.shape} must match x shape {x.shape}.")
    try:
        noise_b = np.broadcast_to(noise, x.shape)
    except ValueError as exc:
        raise ValueError(
            f"noise of shape {noise.shape} is not broadcastable to x shape {x.shape}."
        ) from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(decoded)) or not np.all(np.isfinite(noise_b)):
        raise ValueError("x, noise and decoded must all be finite.")

    if corruption == "additive":
        x_tilde = x + noise_b
    elif corruption == "dropout":
        if not np.all((noise_b == 0) | (noise_b == 1)):
            raise ValueError("with corruption='dropout', noise must be a 0/1 keep-mask.")
        x_tilde = x * noise_b
    else:
        raise ValueError(
            f"corruption must be 'additive' or 'dropout', got {corruption!r}."
        )

    resid = x - decoded
    per_sample = np.sum(resid**2, axis=1)
    loss = float(per_sample.mean())
    corrupt_err = float(np.mean(np.sum((x - x_tilde) ** 2, axis=1)))
    sig = float(np.mean(np.sum(x**2, axis=1)))
    snr = float("inf") if corrupt_err == 0 else 10.0 * float(np.log10(sig / corrupt_err))

    return RichResult(
        title="Denoising autoencoder",
        summary_lines=[("Loss", loss), ("Noise energy", corrupt_err)],
        interpretation=(
            "denoising_gain = noise_energy / loss; above 1 means the network "
            "removed more corruption than it introduced error."
        ),
        payload={
            "loss": loss,
            "mse_per_element": float(np.mean(resid**2)),
            "per_sample_loss": per_sample.tolist(),
            "x_tilde": x_tilde.tolist(),
            "noise_energy": corrupt_err,
            "denoising_gain": float("inf") if loss == 0 else corrupt_err / loss,
            "snr_db": snr,
            "corruption": corruption,
            "estimate": loss,
            "n": int(x.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdae: denoising AE -- corrupt to x_tilde, score ||x - Dec(Enc(x_tilde))||^2 against clean x"
