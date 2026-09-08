# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Autoencoder reconstruction MSE loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_autoencoder_reconstruction_loss"]

_METHOD = "Autoencoder reconstruction loss"


def geron_autoencoder_reconstruction_loss(X, encoded, decoded):
    r"""Squared reconstruction error of an autoencoder.

    .. math::
        L_{\text{AE}} = \frac{1}{m}\sum_{i=1}^{m}
        \bigl\| x_i - \text{Dec}(\text{Enc}(x_i)) \bigr\|_2^2

    The compression ratio is reported alongside: a reconstruction loss is
    only interesting relative to how much the code squeezed the input.
    An autoencoder whose code is as wide as its input can hit zero loss
    by learning the identity and has learned nothing.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Inputs. A 1-D array is treated as a single sample.
    encoded : array-like, shape (m, k)
        Codes produced by the encoder. Used for the compression ratio
        and code statistics, not for the loss itself.
    decoded : array-like, shape (m, d)
        Reconstructions, same shape as ``X``.

    Returns
    -------
    RichResult
        Payload keys ``loss`` (mean squared L2 norm per sample),
        ``mse_per_element``, ``per_sample_loss``, ``code_dim``,
        ``input_dim``, ``compression_ratio``, ``explained_variance``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 18, Autoencoders section.

    Examples
    --------
    >>> r = geron_autoencoder_reconstruction_loss([[1.0, 2.0]], [[0.5]],
    ...                                           [[1.0, 3.0]])
    >>> round(r["loss"], 6)
    1.0
    >>> round(r["mse_per_element"], 6)
    0.5
    >>> r["compression_ratio"]
    2.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    decoded = np.atleast_2d(np.asarray(decoded, dtype=float))
    encoded = np.atleast_2d(np.asarray(encoded, dtype=float))
    if X.shape != decoded.shape:
        raise ValueError(
            f"decoded shape {decoded.shape} must match X shape {X.shape}."
        )
    if X.size == 0:
        raise ValueError("X is empty.")
    if encoded.shape[0] != X.shape[0]:
        raise ValueError(
            f"encoded has {encoded.shape[0]} rows but X has {X.shape[0]}."
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(decoded)):
        raise ValueError("X and decoded must be finite.")

    resid = X - decoded
    per_sample = np.sum(resid**2, axis=1)
    loss = float(per_sample.mean())
    mse_elem = float(np.mean(resid**2))
    var = float(np.sum((X - X.mean(axis=0)) ** 2))
    ev = float("nan") if var == 0 else 1.0 - float(np.sum(resid**2)) / var

    return RichResult(
        title="Autoencoder reconstruction loss",
        summary_lines=[("Loss (per sample)", loss), ("MSE (per element)", mse_elem)],
        payload={
            "loss": loss,
            "mse_per_element": mse_elem,
            "per_sample_loss": per_sample.tolist(),
            "code_dim": int(encoded.shape[1]),
            "input_dim": int(X.shape[1]),
            "compression_ratio": float(X.shape[1] / encoded.shape[1]),
            "explained_variance": ev,
            "estimate": loss,
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grael: autoencoder reconstruction loss ||x - Dec(Enc(x))||^2"
