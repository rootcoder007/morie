# morie.fn -- function file (rootcoder007/morie)
r"""VQ-GAN's encoder: a codebook of context-rich visual parts.

Transformers have no locality prior, so they must learn every
relationship -- expressive, but quadratic in sequence length, which
makes megapixel images infeasible. The fix is not a cheaper attention
but a **shorter sequence**: use a convolutional encoder to compress an
image into a grid of discrete indices into a learned codebook, and let
the transformer model the composition of those parts instead of
pixels.

**Quantisation is nearest neighbour, and it is not differentiable.**
Each encoder output :math:`\hat z_{ij}` is replaced by the closest
codebook entry, and gradients are carried across the gap by a
**straight-through estimator** -- the decoder's gradient is copied
unchanged to the encoder. So the whole thing trains end to end even
though the forward pass contains an ``argmin``.

**The loss has three parts, and the stop-gradients decide who learns
what**:

.. math:: L_{VQ} = \|x - \hat x\|^2
          + \|\mathrm{sg}[E(x)] - z_q\|_2^2
          + \|\mathrm{sg}[z_q] - E(x)\|_2^2,

the second term moving the *codebook* toward the encoder and the third
-- the **commitment loss** -- moving the *encoder* toward its code. Drop
the commitment term and the encoder is free to run away from a
codebook that can never catch it.

**And the compression is the point.** A :math:`256\times256` image at
a downsampling factor of 16 becomes a :math:`16\times16 = 256`-token
sequence, which is what brings a transformer into range at all.

References
----------
Esser, P., Rombach, R. & Ommer, B. (2021) "Taming Transformers for
High-Resolution Image Synthesis", *Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR 2021)*,
12873-12883, arXiv:2012.09841. Sec. 3.1: transformers containing no
inductive prior on locality and therefore having to learn all
relationships, with quadratically increasing cost; the use of a
convolutional approach to learn a codebook of context-rich visual
parts and a transformer to model their global composition; the
nearest-code quantisation s_ij = k such that (z_q)_ij = z_k; the
straight-through gradient estimator copying gradients from the decoder
to the encoder so model and codebook train end to end; and the loss
L_VQ = ||x - x_hat||^2 + ||sg[E(x)] - z_q||^2 + ||sg[z_q] - E(x)||^2
whose last term is the commitment loss.

van den Oord, A., Vinyals, O. & Kavukcuoglu, K. (2017) "Neural
Discrete Representation Learning", *NIPS 2017*, 6306-6315,
arXiv:1711.00937. VQ-VAE, the commitment loss and the
straight-through estimator this builds on.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["quantize", "straight_through", "codebook_loss",
           "commitment_loss", "sequence_length", "encode"]

_EPS = 1e-12


def quantize(vectors, codebook):
    r"""Nearest codebook entry, and its index.

    :math:`s_{ij} = k` such that :math:`(z_q)_{ij} = z_k`.
    """
    Z = [[float(v) for v in r] for r in k.mat(codebook)]
    V = [[float(v) for v in r] for r in k.mat(vectors)]
    if not Z:
        raise ValueError("vqgenc: the codebook is empty")
    if len(Z[0]) != len(V[0]):
        raise ValueError("vqgenc: codebook entries are %d-wide but "
                         "the encoder output is %d"
                         % (len(Z[0]), len(V[0])))
    idx, codes, dists = [], [], []
    for v in V:
        d = [sum((v[a] - z[a]) ** 2 for a in range(len(v)))
             for z in Z]
        j = min(range(len(d)), key=lambda i: d[i])
        idx.append(j)
        codes.append(list(Z[j]))
        dists.append(math.sqrt(d[j]))
    used = len(set(idx))
    return {"indices": idx, "codes": codes, "distance": dists,
            "codebook_size": len(Z), "used": used,
            "usage_fraction": used / float(len(Z)),
            "note": "an argmin, hence not differentiable -- see "
                    "straight_through"}


def straight_through(encoder_output, quantized, upstream_gradient):
    r"""Copy the decoder's gradient to the encoder, unchanged.

    Forward: the quantised code. Backward: identity. That asymmetry is
    what lets an ``argmin`` sit inside an end-to-end model.
    """
    e = [float(v) for v in k.vec(encoder_output)]
    q = [float(v) for v in k.vec(quantized)]
    g = [float(v) for v in k.vec(upstream_gradient)]
    if not (len(e) == len(q) == len(g)):
        raise ValueError("vqgenc: the encoder output, code and "
                         "gradient differ in length")
    return {"forward": list(q), "backward": list(g),
            "jacobian_is_identity": True,
            "note": "forward passes the CODE, backward passes the "
                    "gradient through as if quantisation were absent"}


def codebook_loss(encoder_output, quantized):
    r""":math:`\|\mathrm{sg}[E(x)] - z_q\|^2` -- moves the CODEBOOK."""
    e = [float(v) for v in k.vec(encoder_output)]
    q = [float(v) for v in k.vec(quantized)]
    if len(e) != len(q):
        raise ValueError("vqgenc: the vectors differ in length")
    return {"loss": sum((e[i] - q[i]) ** 2 for i in range(len(e))),
            "gradient_flows_to": "codebook",
            "note": "sg on the encoder side, so only z_q moves"}


def commitment_loss(encoder_output, quantized, beta=0.25):
    r""":math:`\beta\|\mathrm{sg}[z_q] - E(x)\|^2` -- moves the ENCODER.

    Without it the encoder can drift away from a codebook that never
    catches up.
    """
    e = [float(v) for v in k.vec(encoder_output)]
    q = [float(v) for v in k.vec(quantized)]
    if len(e) != len(q):
        raise ValueError("vqgenc: the vectors differ in length")
    b = float(beta)
    if b < 0.0:
        raise ValueError("vqgenc: beta cannot be negative")
    return {"loss": b * sum((e[i] - q[i]) ** 2
                            for i in range(len(e))),
            "beta": b, "gradient_flows_to": "encoder",
            "note": "sg on the code side, so only E(x) moves"}


def sequence_length(height, width, downsample=16):
    r"""The compression that brings a transformer into range."""
    H, W, f = int(height), int(width), int(downsample)
    if f < 1 or H % f or W % f:
        raise ValueError("vqgenc: %dx%d is not divisible by the "
                         "downsampling factor %d" % (H, W, f))
    n = (H // f) * (W // f)
    return {"tokens": n, "pixels": H * W,
            "compression": (H * W) / float(n),
            "attention_cost_pixels": (H * W) ** 2,
            "attention_cost_tokens": n * n,
            "speedup": ((H * W) ** 2) / float(n * n),
            "note": "attention is quadratic, so the saving is the "
                    "SQUARE of the compression"}


def encode(vectors, codebook, beta=0.25, target=None):
    r"""Quantise a grid of encoder outputs and report the VQ loss."""
    q = quantize(vectors, codebook)
    V = [[float(v) for v in r] for r in k.mat(vectors)]
    cb = sum(codebook_loss(V[i], q["codes"][i])["loss"]
             for i in range(len(V)))
    cm = sum(commitment_loss(V[i], q["codes"][i], beta)["loss"]
             for i in range(len(V)))
    rec = 0.0
    if target is not None:
        T = [[float(v) for v in r] for r in k.mat(target)]
        rec = sum((T[i][a] - q["codes"][i][a]) ** 2
                  for i in range(len(T)) for a in range(len(T[0])))
    return RichResult(payload={
        "estimate": q["indices"], "indices": q["indices"],
        "codes": q["codes"], "codebook_loss": cb,
        "commitment_loss": cm, "reconstruction": rec,
        "loss": rec + cb + cm, "usage_fraction": q["usage_fraction"],
        "method": "VQ-GAN encoder and codebook; Esser, Rombach & "
                  "Ommer (2021)",
        "note": "three terms; the stop-gradients decide whether the "
                "codebook or the encoder moves",
    })


def cheatsheet():
    return ("vqgenc: transformers have no locality prior and cost "
            "O(n^2), so shorten the SEQUENCE rather than cheapen the "
            "attention -- a convolutional encoder compresses the image "
            "to a grid of indices into a learned CODEBOOK of visual "
            "parts. Quantisation is nearest-neighbour and NOT "
            "differentiable, so gradients cross by a STRAIGHT-THROUGH "
            "estimator (forward the code, backward the identity). The "
            "loss is reconstruction + ||sg[E(x)] - z_q||^2 (moves the "
            "CODEBOOK) + ||sg[z_q] - E(x)||^2 (the COMMITMENT loss, "
            "moves the ENCODER). Compression f=16 turns 256x256 into "
            "256 tokens; the attention saving is its SQUARE.")


# compact alias per ledger/NAMING.md
vqganencoder = encode

# public names resolved by fn/_lazy_map.json
vqgan_encode = encode
vqganencode = encode
