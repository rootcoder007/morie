# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DCGAN generator: transposed-conv upsampling from latent z."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dcgan_generator"]

_METHOD = "DCGAN generator forward pass"


def _conv_transpose2d(Z, K, s):
    kh, kw = K.shape
    h, w = Z.shape
    out = np.zeros(((h - 1) * s + kh, (w - 1) * s + kw), dtype=float)
    for i in range(h):
        for j in range(w):
            out[i * s:i * s + kh, j * s:j * s + kw] += Z[i, j] * K
    return out


def _batchnorm(A, eps=1e-5):
    mu = float(A.mean())
    sd = float(A.std())
    if sd < np.sqrt(eps):
        return A - mu
    return (A - mu) / np.sqrt(sd**2 + eps)


def geron_dcgan_generator(z, weights, seed_shape=None, stride=2, batch_norm=True):
    r"""Map a latent vector to an image through transposed convolutions.

    .. math::
        G(z) = \text{conv\_transpose\_stack}(W_{\text{proj}} z)

    ``weights[0]`` is the projection that reshapes the latent into a
    small spatial seed; ``weights[1:]`` are transposed-convolution
    kernels, each doubling the spatial size at ``stride=2``.  Following
    the DCGAN recipe every layer but the last gets batch norm and ReLU,
    and the output goes through ``tanh`` -- which is why DCGAN training
    data is scaled to ``[-1, 1]`` rather than ``[0, 1]``.

    Parameters
    ----------
    z : array-like, shape (k,)
        Latent vector.
    weights : sequence
        ``weights[0]``: projection matrix of shape ``(k, h0*w0)``.
        ``weights[1:]``: 2-D transposed-conv kernels, at least one.
    seed_shape : tuple, optional
        ``(h0, w0)``; inferred as a square if omitted.
    stride : int, optional
        Upsampling stride per layer, default 2.
    batch_norm : bool, optional
        Apply batch norm before the ReLU of each hidden layer.

    Returns
    -------
    RichResult
        Payload keys ``image``, ``image_shape``, ``seed``,
        ``layer_shapes``, ``upsample_factor``, ``estimate`` (mean pixel
        value), ``n``, ``method``.

    References
    ----------
    Géron Ch 18, DCGAN section.

    Examples
    --------
    A latent of length 1 projected to a ``2x2`` seed of ones, then
    upsampled once by a ``2x2`` kernel of ones at stride 2: every output
    pixel is ``tanh(1)``.

    >>> W0 = [[1.0, 1.0, 1.0, 1.0]]
    >>> K1 = [[1.0, 1.0], [1.0, 1.0]]
    >>> r = geron_dcgan_generator([1.0], [W0, K1], seed_shape=(2, 2))
    >>> r["image_shape"]
    (4, 4)
    >>> round(r["image"][0][0], 6)
    0.761594
    >>> len(set(round(v, 12) for row in r["image"] for v in row))
    1
    """
    z = np.asarray(z, dtype=float).ravel()
    if z.size == 0:
        raise ValueError("z is empty.")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values.")
    ws = list(weights)
    if len(ws) < 2:
        raise ValueError(
            "weights must hold a projection matrix followed by at least one "
            f"transposed-conv kernel, got {len(ws)} entries."
        )
    W0 = np.atleast_2d(np.asarray(ws[0], dtype=float))
    if W0.shape[0] != z.size:
        raise ValueError(
            f"projection matrix has {W0.shape[0]} rows but z has {z.size} entries."
        )
    if not np.all(np.isfinite(W0)):
        raise ValueError("the projection matrix contains non-finite values.")
    flat = z @ W0
    if seed_shape is None:
        side = int(round(np.sqrt(flat.size)))
        if side * side != flat.size:
            raise ValueError(
                f"projection gives {flat.size} units, which is not a perfect square; "
                "pass seed_shape=(h0, w0)."
            )
        seed_shape = (side, side)
    h0, w0 = (int(v) for v in seed_shape)
    if h0 < 1 or w0 < 1 or h0 * w0 != flat.size:
        raise ValueError(
            f"seed_shape {(h0, w0)} does not match the {flat.size} projected units."
        )
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}.")

    A = flat.reshape(h0, w0)
    seed = A.copy()
    shapes = [(h0, w0)]
    kernels = ws[1:]
    for i, K in enumerate(kernels):
        K = np.atleast_2d(np.asarray(K, dtype=float))
        if K.ndim != 2 or K.size == 0:
            raise ValueError(f"weights[{i + 1}] must be a non-empty 2-D kernel.")
        if not np.all(np.isfinite(K)):
            raise ValueError(f"weights[{i + 1}] contains non-finite values.")
        A = _conv_transpose2d(A, K, stride)
        if i < len(kernels) - 1:
            if batch_norm:
                A = _batchnorm(A)
            A = np.maximum(A, 0.0)
        shapes.append((int(A.shape[0]), int(A.shape[1])))
    image = np.tanh(A)

    return RichResult(
        title="DCGAN generator",
        summary_lines=[("Output shape", tuple(int(v) for v in image.shape)),
                       ("Layers", len(kernels))],
        payload={
            "image": image.tolist(),
            "image_shape": (int(image.shape[0]), int(image.shape[1])),
            "seed": seed.tolist(),
            "layer_shapes": shapes,
            "upsample_factor": float(image.shape[0] / h0),
            "latent_dim": int(z.size),
            "estimate": float(image.mean()),
            "n": int(image.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdcgan: DCGAN generator -- project z to a seed map, transposed-conv upsample, tanh out"
