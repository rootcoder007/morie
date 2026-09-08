# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deep convolutional GAN (DCGAN)."""

from . import _array_core as np

from ._richresult import RichResult
from .grdcgan import geron_dcgan_generator

__all__ = ["geron_dcgan"]


def geron_dcgan(X, z_dim=100, filters=64, epochs=50, lr=0.0002, seed_shape=(4, 4), stride=2):
    """
    Deep convolutional GAN (DCGAN).

    Formula: generator: transposed-conv upsample; discriminator:
    strided-conv down

    This is an architecture specification, so it is resolved against a
    concrete input exactly like ``hmalex`` does: given the image shape in
    ``X``, both networks are laid out layer by layer with real output
    shapes and parameter counts.

    * the generator projects ``z`` to a ``seed_shape`` map with
      ``filters * 2^(L-1)`` channels and then upsamples by ``stride`` per
      transposed-conv layer, halving the channels each time, until the
      image resolution is reached;
    * the discriminator mirrors it with strided convolutions that halve
      the resolution and double the channels, ending in one logit.

    The layer count is not a free choice: ``L = log_stride(H / h0)`` must
    be a whole number, and an image size that does not decompose that way
    is an error rather than a silently rounded layout -- which is the
    DCGAN constraint people actually trip over.

    One generator forward pass is DELEGATED to
    :func:`morie.fn.grdcgan.geron_dcgan_generator` with unit kernels, so
    the resolved shapes are demonstrated on real arithmetic rather than
    only asserted.

    ``epochs`` and ``lr`` are validated and recorded as the training
    configuration (DCGAN's paper values are ``lr = 2e-4`` with Adam
    ``beta1 = 0.5``), but no training is performed here: see ``hmgan``
    for the minimax loop.

    Parameters
    ----------
    X : array-like, shape (H, W) or (m, H, W)
        Real images, used for their shape.
    z_dim : int, default 100
    filters : int, default 64
        Base channel count of the last generator layer.
    epochs : int, default 50
    lr : float, default 0.0002
    seed_shape : tuple, default (4, 4)
        Spatial size of the projected latent.
    stride : int, default 2

    Returns
    -------
    result : RichResult
        Keys: generator_layers, discriminator_layers, generator_params,
        discriminator_params, total_params, image_shape, n_layers,
        sample_shape, training_config, estimate, n, method.

    Examples
    --------
    A 16x16 image from a 4x4 seed at stride 2 needs exactly two
    upsampling layers:

    >>> import numpy as np
    >>> X = np.zeros((3, 16, 16))
    >>> r = geron_dcgan(X, z_dim=8, filters=4)
    >>> r["n_layers"]
    2
    >>> [l["out"] for l in r["generator_layers"] if l["kind"] == "deconv"]
    [8, 16]
    >>> [l["out"] for l in r["discriminator_layers"] if l["kind"] == "conv"]
    [8, 4]

    The parameter counts are exact: the projection is
    ``8 * (4*4*8) + 128 = 1152``.

    >>> r["generator_layers"][0]["params"]
    1152
    >>> r["sample_shape"]
    (16, 16)

    A resolution that is not a power of the stride above the seed is
    rejected:

    >>> geron_dcgan(np.zeros((1, 20, 20)), z_dim=8, filters=4)
    Traceback (most recent call last):
      ...
    ValueError: geron_dcgan: image side 20 is not 4 times a power of 2; DCGAN cannot reach it by stride-2 upsampling

    References
    ----------
    Géron Ch 18
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 2:
        A = A[None, :, :]
    if A.ndim != 3 or A.size == 0:
        raise ValueError(f"geron_dcgan: X must be (H, W) or (m, H, W), got shape {A.shape}")
    m, H, W = A.shape
    if H != W:
        raise ValueError(f"geron_dcgan: DCGAN assumes square images, got {H}x{W}")
    k = int(z_dim)
    if k < 1:
        raise ValueError(f"geron_dcgan: z_dim must be >= 1, got {z_dim!r}")
    f = int(filters)
    if f < 1:
        raise ValueError(f"geron_dcgan: filters must be >= 1, got {filters!r}")
    st = int(stride)
    if st < 2:
        raise ValueError(f"geron_dcgan: stride must be >= 2 for upsampling, got {stride!r}")
    h0, w0 = (int(v) for v in seed_shape)
    if h0 < 1 or w0 < 1:
        raise ValueError(f"geron_dcgan: seed_shape must be positive, got {seed_shape!r}")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_dcgan: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_dcgan: lr must be positive and finite, got {lr!r}")

    ratio = H / h0
    L = int(round(np.log(ratio) / np.log(st))) if ratio >= 1 else -1
    if L < 1 or h0 * st**L != H:
        raise ValueError(
            f"geron_dcgan: image side {H} is not {h0} times a power of {st}; "
            f"DCGAN cannot reach it by stride-{st} upsampling"
        )

    kern = 4
    gen = [{
        "kind": "project",
        "in": k,
        "out": h0,
        "channels": f * st ** (L - 1),
        "params": int(k * (h0 * w0 * f * st ** (L - 1)) + h0 * w0 * f * st ** (L - 1)),
    }]
    size = h0
    ch = f * st ** (L - 1)
    for i in range(L):
        out_ch = 1 if i == L - 1 else ch // st
        size *= st
        gen.append({
            "kind": "deconv",
            "in_channels": ch,
            "channels": out_ch,
            "kernel": kern,
            "stride": st,
            "out": int(size),
            "params": int(out_ch * (kern * kern * ch) + out_ch),
            "batch_norm": i < L - 1,
            "activation": "tanh" if i == L - 1 else "relu",
        })
        ch = out_ch

    dis = []
    size = H
    ch = 1
    for i in range(L):
        out_ch = f if i == 0 else ch * st
        size //= st
        dis.append({
            "kind": "conv",
            "in_channels": ch,
            "channels": out_ch,
            "kernel": kern,
            "stride": st,
            "out": int(size),
            "params": int(out_ch * (kern * kern * ch) + out_ch),
            "batch_norm": i > 0,
            "activation": "leaky_relu",
        })
        ch = out_ch
    flat = int(size * size * ch)
    dis.append({"kind": "fc", "in": flat, "out": 1, "params": flat + 1, "activation": "sigmoid"})

    g_params = int(sum(l["params"] for l in gen))
    d_params = int(sum(l["params"] for l in dis))

    # Demonstrate the resolved shapes on a real forward pass.
    Wproj = np.ones((k, h0 * w0)) / k
    kernels = [np.ones((st, st))] * L
    demo = geron_dcgan_generator(np.ones(k), [Wproj] + kernels, seed_shape=(h0, w0), stride=st)

    return RichResult(
        title="DCGAN architecture",
        summary_lines=[("Image", (H, W)), ("Generator params", g_params), ("Discriminator params", d_params)],
        tables=[{
            "title": "Generator",
            "headers": ["kind", "channels", "out", "params"],
            "rows": [[l["kind"], l.get("channels"), l["out"], l["params"]] for l in gen],
        }],
        interpretation="Both nets are pure convolution: no pooling, no fully-connected hidden layers, which is DCGAN's rule.",
        payload={
            "generator_layers": gen,
            "discriminator_layers": dis,
            "generator_params": g_params,
            "discriminator_params": d_params,
            "total_params": g_params + d_params,
            "image_shape": (int(H), int(W)),
            "n_layers": int(L),
            "z_dim": k,
            "sample_shape": tuple(int(v) for v in demo["image_shape"]),
            "training_config": {"epochs": E, "lr": eta, "adam_beta1": 0.5, "note": "training loop lives in hmgan"},
            "estimate": float(g_params + d_params),
            "n": int(m),
            "method": "DCGAN generator/discriminator resolved to concrete shapes; forward pass delegated to grdcgan",
        },
    )


def cheatsheet():
    return "hmdcg: Deep convolutional GAN (DCGAN)"


# compact alias per ledger/NAMING.md
gerondcgan = geron_dcgan
