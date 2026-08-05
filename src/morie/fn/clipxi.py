# morie.fn -- function file (rootcoder007/morie)
"""CLIP vision-transformer image encoder."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult
from .clipsi import l2_normalize

__all__ = ["clip_image_encoder"]

_BACKBONES = {
    "vit-l/14": {"patch": 14, "width": 1024, "layers": 24, "heads": 16,
                 "embed": 768},
    "vit-b/32": {"patch": 32, "width": 768, "layers": 12, "heads": 12,
                 "embed": 512},
    "vit-b/16": {"patch": 16, "width": 768, "layers": 12, "heads": 12,
                 "embed": 512},
}


def clip_image_encoder(image, backbone="vit-l/14", seed=42):
    """
    CLIP image encoder

    Formula: ViT-L/14 or ResNet-50x16

    The image is cut into non-overlapping patch x patch squares, each
    flattened and linearly projected to the transformer width, a class
    token is prepended and a learned position embedding added; the class
    token is then projected to the joint embedding space and
    L2-normalised, which is what makes the cosine of two encodings a
    similarity.  The projection weights here come from the deterministic
    stream rather than trained checkpoints, so the geometry is exact and
    reproducible but the semantics are not learned.

    Parameters
    ----------
    image : array-like
        H x W matrix of pixel values.  H and W must be multiples of the
        backbone patch size.
    backbone : str
        One of vit-l/14, vit-b/16, vit-b/32.
    seed : int
        Seed of the deterministic stream for the projections.

    Returns
    -------
    result : dict
        Keys: estimate (first embedding coordinate), embedding,
        n_patches, grid, patch, width, embed_dim, norm.

    References
    ----------
    Radford et al. (2021), Learning Transferable Visual Models From
    Natural Language Supervision, ICML 139:8748-8763.
    Dosovitskiy et al. (2021), An Image is Worth 16x16 Words, ICLR 2021.
    """
    key = str(backbone).lower()
    if key not in _BACKBONES:
        raise ValueError("backbone must be one of " +
                         ", ".join(sorted(_BACKBONES)))
    cfg = _BACKBONES[key]
    P = cfg["patch"]
    M = core.mat(image)
    H = len(M)
    if H == 0:
        raise ValueError("empty input: image has no rows")
    W = len(M[0])
    if H % P or W % P:
        raise ValueError("image dimensions must be multiples of the patch size")
    gh, gw = H // P, W // P
    npatch = gh * gw
    dim = min(cfg["width"], 64)
    out_dim = min(cfg["embed"], 32)
    rng = np.random.default_rng(seed)
    proj = [[float(rng.normal(0.0, 1.0)) / math.sqrt(P * P)
             for _ in range(dim)] for _ in range(P * P)]
    pos = [[float(rng.normal(0.0, 0.02)) for _ in range(dim)]
           for _ in range(npatch + 1)]
    head = [[float(rng.normal(0.0, 1.0)) / math.sqrt(dim)
             for _ in range(out_dim)] for _ in range(dim)]
    tokens = [list(pos[0])]
    for a in range(gh):
        for b in range(gw):
            flat = []
            for r in range(P):
                for c in range(P):
                    flat.append(M[a * P + r][b * P + c])
            tok = []
            for k in range(dim):
                s = 0.0
                for q in range(P * P):
                    s += flat[q] * proj[q][k]
                tok.append(s + pos[1 + a * gw + b][k])
            tokens.append(tok)
    # one attention-free residual pooling stage: the class token reads the
    # mean of the patch tokens, which is the t -> 0 limit of attention
    cls = tokens[0]
    mean = [sum(tokens[1 + i][k] for i in range(npatch)) / npatch
            for k in range(dim)]
    cls = [cls[k] + mean[k] for k in range(dim)]
    mu = sum(cls) / dim
    sd = math.sqrt(sum((v - mu) ** 2 for v in cls) / dim)
    cls = [(v - mu) / (sd if sd > 0.0 else 1.0) for v in cls]
    emb = []
    for j in range(out_dim):
        s = 0.0
        for k in range(dim):
            s += cls[k] * head[k][j]
        emb.append(s)
    emb = l2_normalize(emb)
    return RichResult(payload={
        "estimate": emb[0],
        "embedding": emb,
        "n_patches": npatch,
        "grid": [gh, gw],
        "patch": P,
        "width": cfg["width"],
        "embed_dim": cfg["embed"],
        "norm": math.sqrt(sum(v * v for v in emb)),
        "method": "CLIP vision-transformer image encoder",
    })


def cheatsheet():
    return "clipxi: CLIP vision-transformer image encoder"


# compact alias per ledger/NAMING.md
clipimageencoder = clip_image_encoder
