# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convolutional autoencoder: conv encoder + transposed-conv decoder."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_convolutional_autoencoder"]

_METHOD = "Convolutional autoencoder forward pass"


def _conv2d_valid(Z, K, s):
    kh, kw = K.shape
    h, w = Z.shape
    if h < kh or w < kw:
        raise ValueError(
            f"encoder kernel {kh}x{kw} does not fit the {h}x{w} feature map."
        )
    oh = (h - kh) // s + 1
    ow = (w - kw) // s + 1
    out = np.empty((oh, ow), dtype=float)
    for i in range(oh):
        for j in range(ow):
            out[i, j] = float(np.sum(Z[i * s:i * s + kh, j * s:j * s + kw] * K))
    return out


def _conv_transpose2d(Z, K, s):
    kh, kw = K.shape
    h, w = Z.shape
    out = np.zeros(((h - 1) * s + kh, (w - 1) * s + kw), dtype=float)
    for i in range(h):
        for j in range(w):
            out[i * s:i * s + kh, j * s:j * s + kw] += Z[i, j] * K
    return out


_OUT_ACTS = {"identity": lambda a: a, "tanh": np.tanh,
             "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a))}


def geron_convolutional_autoencoder(x, encoder_weights, decoder_weights,
                                    stride=2, output_activation="identity"):
    r"""Encode an image with strided convolutions and decode it back.

    .. math::
        z = \text{ConvEnc}(x), \qquad \hat x = \text{DeConv}(z),
        \qquad L = \|x - \hat x\|^2

    Downsampling by *stride* rather than pooling is the convolutional
    autoencoder's usual choice, and the decoder undoes it with
    transposed convolution -- which scatters each code cell over a
    kernel-sized patch and sums the overlaps.  The output shape follows
    :math:`(n-1)s + k` per axis, so the kernel and stride must be chosen
    to land back on the input size; this routine raises rather than
    silently cropping if they do not.

    Parameters
    ----------
    x : array-like, shape (H, W)
        Single-channel input image.
    encoder_weights : sequence of 2-D array-like
        Convolution kernels applied in order, each at ``stride``, with a
        ReLU after every layer.
    decoder_weights : sequence of 2-D array-like
        Transposed-convolution kernels applied in order, each at
        ``stride``, with a ReLU after every layer but the last.
    stride : int, optional
        Shared stride, default 2.
    output_activation : {"identity", "tanh", "sigmoid"}, optional
        Applied to the reconstruction.

    Returns
    -------
    RichResult
        Payload keys ``x_hat``, ``code``, ``code_shape``, ``loss``,
        ``mse_per_pixel``, ``compression_ratio``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 18, Convolutional Autoencoders section.

    Examples
    --------
    A ``1x1`` encoder kernel at stride 2 keeps the top-left pixel of a
    ``2x2`` image; a ``2x2`` decoder kernel of ones broadcasts it back
    over the whole image:

    >>> r = geron_convolutional_autoencoder([[1.0, 2.0], [3.0, 4.0]],
    ...                                     [[[1.0]]],
    ...                                     [[[1.0, 1.0], [1.0, 1.0]]])
    >>> r["code"]
    [[1.0]]
    >>> r["x_hat"]
    [[1.0, 1.0], [1.0, 1.0]]
    >>> r["loss"]
    14.0
    >>> r["compression_ratio"]
    4.0
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.ndim != 2 or x.size == 0:
        raise ValueError(f"x must be a non-empty 2-D image, got shape {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")
    enc = [np.atleast_2d(np.asarray(K, dtype=float)) for K in encoder_weights]
    dec = [np.atleast_2d(np.asarray(K, dtype=float)) for K in decoder_weights]
    if not enc:
        raise ValueError("encoder_weights is empty.")
    if not dec:
        raise ValueError("decoder_weights is empty.")
    for tag, ks in (("encoder", enc), ("decoder", dec)):
        for i, K in enumerate(ks):
            if K.ndim != 2 or K.size == 0:
                raise ValueError(f"{tag}_weights[{i}] must be a non-empty 2-D kernel.")
            if not np.all(np.isfinite(K)):
                raise ValueError(f"{tag}_weights[{i}] contains non-finite values.")
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}.")
    if output_activation not in _OUT_ACTS:
        raise ValueError(
            f"output_activation must be one of {sorted(_OUT_ACTS)}, got "
            f"{output_activation!r}."
        )

    z = x
    for K in enc:
        z = np.maximum(_conv2d_valid(z, K, stride), 0.0)
    code = z

    a = code
    for i, K in enumerate(dec):
        a = _conv_transpose2d(a, K, stride)
        if i < len(dec) - 1:
            a = np.maximum(a, 0.0)
    x_hat = _OUT_ACTS[output_activation](a)

    if x_hat.shape != x.shape:
        raise ValueError(
            f"the decoder produced a {x_hat.shape} reconstruction but the input is "
            f"{x.shape}; transposed conv gives (n-1)*s + k per axis, so adjust the "
            "decoder kernel sizes or the stride."
        )

    resid = x - x_hat
    loss = float(np.sum(resid**2))

    return RichResult(
        title="Convolutional autoencoder",
        summary_lines=[("Loss", loss), ("Code shape", tuple(int(v) for v in code.shape))],
        payload={
            "x_hat": x_hat.tolist(),
            "code": code.tolist(),
            "code_shape": (int(code.shape[0]), int(code.shape[1])),
            "loss": loss,
            "mse_per_pixel": float(np.mean(resid**2)),
            "compression_ratio": float(x.size / code.size),
            "estimate": loss,
            "n": int(x.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grcae: conv autoencoder -- strided conv encoder, transposed-conv decoder, ||x-x_hat||^2"
