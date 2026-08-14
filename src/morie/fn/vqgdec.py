# morie.fn -- function file (rootcoder007/morie)
r"""VQ-GAN's decoder: perceptual quality at high compression.

The encoder side (:mod:`vqgenc`) buys a short sequence. The decoder
side has to pay for it: at a compression factor of 16 an
:math:`L_2` reconstruction loss produces blur, because :math:`L_2` is
minimised by the *conditional mean* of every image consistent with the
code. VQ-GAN replaces it with a **perceptual loss** and adds an
**adversarial** term from a **patch-based** discriminator, so the
codebook has to capture perceptually important local structure rather
than average it away.

**The adversarial weight is computed, not tuned.** The paper sets

.. math:: \lambda = \frac{\nabla_{G_L}[L_{rec}]}
                        {\nabla_{G_L}[L_{GAN}] + \delta},

with gradients taken with respect to the **last layer** of the
decoder. So the two losses arrive at the final layer with matched
magnitudes automatically, whatever their scales -- and when the GAN
gradient explodes, :math:`\lambda` shrinks rather than the training
collapsing.

**Decoding is exact where it needs to be.** Indices map to codes by
lookup, so ``decode_indices`` inverts ``quantize`` exactly on any
vector already in the codebook -- an identity the anchor checks rather
than approximating.

**Sliding-window generation** produces images larger than the
transformer's context: the model is applied patch-wise across the
latent grid. That is valid as long as the dataset statistics are
roughly spatially invariant or spatial conditioning is available --
and when it is not, the paper's own remedy is to condition on image
coordinates.

References
----------
Esser, P., Rombach, R. & Ommer, B. (2021) "Taming Transformers for
High-Resolution Image Synthesis", *CVPR 2021*, 12873-12883,
arXiv:2012.09841. Sec. 3.1 ("Learning a Perceptually Rich Codebook":
replacing the L2 reconstruction loss with a perceptual loss and
introducing adversarial training with a PATCH-BASED discriminator to
keep perceptual quality at increased compression), Sec. 3.2 (the
adaptive weight lambda = grad_GL[L_rec] / (grad_GL[L_GAN] + delta)
computed with respect to the last layer L of the decoder), and Sec. 3.3
(sliding-window generation, valid when the dataset statistics are
approximately spatially invariant or spatial conditioning is
available, with conditioning on image coordinates as the remedy
otherwise).

Isola, P., Zhu, J.-Y., Zhou, T. & Efros, A. A. (2017) "Image-to-Image
Translation with Conditional Adversarial Networks", *CVPR 2017*,
1125-1134, arXiv:1611.07004. The patch-based discriminator.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. (2018)
"The Unreasonable Effectiveness of Deep Features as a Perceptual
Metric", *CVPR 2018*, 586-595, arXiv:1801.03924. The perceptual loss.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["decode_indices", "adaptive_weight", "patch_discriminator",
           "sliding_windows", "decode"]

_EPS = 1e-12


def decode_indices(indices, codebook):
    r"""Lookup. Exactly inverts quantisation for in-codebook vectors."""
    Z = [[float(v) for v in r] for r in k.mat(codebook)]
    out = []
    for i in indices:
        j = int(i)
        if j < 0 or j >= len(Z):
            raise ValueError("vqgdec: index %d is outside a codebook "
                             "of %d" % (j, len(Z)))
        out.append(list(Z[j]))
    return {"codes": out, "n": len(out),
            "note": "exact: the index IS the code"}


def adaptive_weight(grad_rec, grad_gan, delta=1e-6, clip=1e4):
    r""":math:`\lambda = \nabla_{G_L}[L_{rec}] /
    (\nabla_{G_L}[L_{GAN}] + \delta)`.

    Balances the two losses at the decoder's LAST layer, so their raw
    scales stop mattering; a runaway GAN gradient shrinks lambda
    instead of destroying the reconstruction.
    """
    gr = abs(float(grad_rec))
    gg = abs(float(grad_gan))
    d = float(delta)
    if d <= 0.0:
        raise ValueError("vqgdec: delta must be positive")
    lam = gr / (gg + d)
    return {"lambda": min(lam, float(clip)), "raw": lam,
            "clipped": lam > float(clip),
            "note": "gradients taken w.r.t. the LAST layer of the "
                    "decoder"}


def patch_discriminator(image, patch=4, scorer=None):
    r"""Score PATCHES, not the whole image.

    A whole-image verdict is one scalar for millions of pixels; a
    patch discriminator gives a dense signal about local texture,
    which is what the codebook needs to learn.
    """
    I = [[float(v) for v in r] for r in k.mat(image)]
    p = int(patch)
    H, W = len(I), len(I[0])
    if p < 1 or H % p or W % p:
        raise ValueError("vqgdec: a %dx%d image does not tile into "
                         "%dx%d patches" % (H, W, p, p))
    scores = []
    for i in range(0, H, p):
        row = []
        for j in range(0, W, p):
            blk = [I[a][b] for a in range(i, i + p)
                   for b in range(j, j + p)]
            row.append(float(scorer(blk)) if scorer is not None
                       else sum(blk) / len(blk))
        scores.append(row)
    flat = [v for r in scores for v in r]
    return {"scores": scores, "n_patches": len(flat),
            "mean": sum(flat) / len(flat),
            "note": "one verdict per patch, not one per image"}


def sliding_windows(height, width, window, stride=None):
    r"""Generate an image larger than the transformer's context.

    Valid when the dataset statistics are roughly spatially invariant
    or spatial conditioning is available -- otherwise condition on
    image coordinates.
    """
    H, W, w = int(height), int(width), int(window)
    s = w if stride is None else int(stride)
    if w < 1 or s < 1 or w > H or w > W:
        raise ValueError("vqgdec: the window must fit inside the "
                         "latent grid")
    wins = []
    i = 0
    while i + w <= H:
        j = 0
        while j + w <= W:
            wins.append((i, j))
            j += s
        if j - s + w < W:
            wins.append((i, W - w))
        i += s
    if i - s + w < H:
        j = 0
        while j + w <= W:
            wins.append((H - w, j))
            j += s
        wins.append((H - w, W - w))
    covered = set()
    for (a, b) in wins:
        for x in range(a, a + w):
            for y in range(b, b + w):
                covered.add((x, y))
    return {"windows": sorted(set(wins)),
            "n_windows": len(set(wins)),
            "covers_everything": len(covered) == H * W,
            "context": w * w,
            "note": "spatially invariant statistics, or condition on "
                    "coordinates"}


def decode(indices, codebook, generator=None, grad_rec=None,
           grad_gan=None):
    r"""Indices to codes to image, with the adaptive GAN weight."""
    d = decode_indices(indices, codebook)
    img = generator(d["codes"]) if generator is not None \
        else d["codes"]
    lam = None
    if grad_rec is not None and grad_gan is not None:
        lam = adaptive_weight(grad_rec, grad_gan)["lambda"]
    return RichResult(payload={
        "estimate": img, "image": img, "codes": d["codes"],
        "n_tokens": d["n"], "adaptive_lambda": lam,
        "method": "VQ-GAN decoder with perceptual and adversarial "
                  "losses; Esser, Rombach & Ommer (2021)",
        "note": "L2 at this compression gives blur, because it "
                "returns the conditional MEAN of every consistent "
                "image",
    })


def cheatsheet():
    return ("vqgdec: at compression 16 an L2 loss returns the "
            "conditional MEAN of every image consistent with the code, "
            "which is blur. So use a PERCEPTUAL loss plus an "
            "adversarial term from a PATCH-BASED discriminator -- one "
            "verdict per patch, not per image. The adversarial weight "
            "is COMPUTED, not tuned: lambda = grad[L_rec] / "
            "(grad[L_GAN] + delta) at the decoder's LAST layer, so a "
            "runaway GAN gradient shrinks lambda instead of wrecking "
            "training. Sliding-window generation exceeds the context, "
            "valid under spatially invariant statistics or coordinate "
            "conditioning.")


# compact alias per ledger/NAMING.md
vqgandecoder = decode

# public names resolved by fn/_lazy_map.json
vqgan_decode = decode
vqgandecode = decode
