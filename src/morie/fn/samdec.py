# morie.fn -- function file (rootcoder007/morie)
r"""SAM's mask decoder: two-way attention, then a dynamic classifier.

The decoder maps an image embedding, the prompt embeddings and an
extra learned **output token** to a mask, cheaply enough to run per
prompt in real time.

**Attention runs in both directions.** A modified Transformer decoder
block applies prompt self-attention, then cross-attention
prompt-to-image *and* image-to-prompt, updating **both** sets of
embeddings. One direction alone would let the prompt read the image
without the image ever being told what was asked.

**Two blocks, then upsample.** The image embedding is upsampled and an
MLP maps the output token to a **dynamic linear classifier** -- the
weights of the final classifier are produced by the network from the
prompt, so the mask is a spatially-dense dot product against a vector
that depends on what was asked. This is what makes one decoder answer
different prompts without any per-prompt training.

**The loss is focal plus dice.** Focal loss down-weights easy pixels
by :math:`(1-p_t)^\gamma`, which matters because a mask is
overwhelmingly background; dice loss scores overlap directly, so it is
insensitive to that same imbalance. ``focal_loss`` and ``dice_loss``
are separate here, and the anchor checks the down-weighting factor
against its closed form rather than trusting the name.

References
----------
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson,
L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P. &
Girshick, R. (2023) "Segment Anything", *ICCV 2023*, 4015-4026,
arXiv:2304.02643. Sec. 3 and Appendix A: the mask decoder mapping the
image embedding, prompt embeddings and an output token to a mask; the
modified Transformer decoder block using prompt self-attention and
cross-attention in TWO directions to update all embeddings; two such
blocks followed by upsampling the image embedding and an MLP mapping
the output token to a dynamic linear classifier that computes the
mask foreground probability at each location; and supervision by a
linear combination of focal loss and dice loss.

Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollar, P. (2017)
"Focal Loss for Dense Object Detection", *ICCV 2017*, 2980-2988,
arXiv:1708.02002. The (1-p_t)^gamma down-weighting.

Milletari, F., Navab, N. & Ahmadi, S.-A. (2016) "V-Net: Fully
Convolutional Neural Networks for Volumetric Medical Image
Segmentation", *3DV 2016*, 565-571, arXiv:1606.04797. The dice loss.

Vaswani, A. et al. (2017) "Attention Is All You Need", *NIPS 2017*,
5998-6008, arXiv:1706.03762. The decoder block being modified.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["two_way_block", "upsample", "dynamic_mask_head",
           "focal_loss", "dice_loss", "decode_mask"]

_EPS = 1e-12


def _attend(Q, K, V):
    d = len(Q[0])
    out, W = [], []
    for q in Q:
        sc = [sum(q[a] * kk[a] for a in range(d)) / math.sqrt(d)
              for kk in K]
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        W.append(w)
        out.append([sum(w[j] * V[j][a] for j in range(len(V)))
                    for a in range(len(V[0]))])
    return out, W


def two_way_block(prompt_tokens, image_tokens):
    r"""Prompt self-attention, then cross-attention BOTH ways.

    Both sets of embeddings are updated: the prompt learns what the
    image contains, and the image learns what was asked of it.
    """
    P = [[float(v) for v in r] for r in k.mat(prompt_tokens)]
    I = [[float(v) for v in r] for r in k.mat(image_tokens)]
    if len(P[0]) != len(I[0]):
        raise ValueError("samdec: prompt tokens are %d-dimensional "
                         "but image tokens are %d"
                         % (len(P[0]), len(I[0])))
    sa, _ = _attend(P, P, P)
    P1 = [[P[i][a] + sa[i][a] for a in range(len(P[0]))]
          for i in range(len(P))]
    p2i, w_p2i = _attend(P1, I, I)
    P2 = [[P1[i][a] + p2i[i][a] for a in range(len(P1[0]))]
          for i in range(len(P1))]
    i2p, w_i2p = _attend(I, P2, P2)
    I2 = [[I[i][a] + i2p[i][a] for a in range(len(I[0]))]
          for i in range(len(I))]
    return {"prompt_tokens": P2, "image_tokens": I2,
            "prompt_to_image": w_p2i, "image_to_prompt": w_i2p,
            "note": "both directions, so both embeddings move"}


def upsample(grid, factor=2):
    r"""Nearest-neighbour upsampling of the image embedding."""
    G = [[float(v) for v in r] for r in k.mat(grid)]
    f = int(factor)
    if f < 1:
        raise ValueError("samdec: the upsampling factor must be >= 1")
    out = []
    for i in range(len(G) * f):
        out.append([G[i // f][j // f] for j in range(len(G[0]) * f)])
    return out


def dynamic_mask_head(output_token, image_grid_vectors, mlp=None):
    r"""The output token BECOMES the classifier weights.

    The mask is a dot product of each spatial vector with a weight
    vector produced from the prompt, which is how one decoder answers
    prompts it was never trained on individually.
    """
    w = [float(v) for v in k.vec(output_token)]
    if mlp is not None:
        w = [float(v) for v in k.vec(mlp(w))]
    G = image_grid_vectors
    H, W = len(G), len(G[0])
    d = len(G[0][0])
    if len(w) != d:
        raise ValueError("samdec: the dynamic classifier is %d-wide "
                         "but the spatial vectors are %d"
                         % (len(w), d))
    logits = [[sum(w[a] * G[i][j][a] for a in range(d))
               for j in range(W)] for i in range(H)]
    return {"logits": logits,
            "probability": [[1.0 / (1.0 + math.exp(-min(60.0,
                                                        max(-60.0, v))))
                             for v in row] for row in logits],
            "weights": w,
            "note": "the classifier weights come from the PROMPT"}


def focal_loss(prob, target, gamma=2.0, alpha=0.25):
    r""":math:`-\alpha_t (1-p_t)^\gamma \log p_t`.

    A mask is mostly background, so an unweighted cross entropy is
    dominated by pixels that are already right.
    """
    p = [float(v) for v in k.vec(prob)]
    t = [float(v) for v in k.vec(target)]
    if len(p) != len(t):
        raise ValueError("samdec: the prediction and target differ "
                         "in size")
    g, a = float(gamma), float(alpha)
    tot, mods = 0.0, []
    for i in range(len(p)):
        pt = p[i] if t[i] > 0.5 else 1.0 - p[i]
        at = a if t[i] > 0.5 else 1.0 - a
        mod = (1.0 - pt) ** g
        mods.append(mod)
        tot += -at * mod * math.log(max(pt, _EPS))
    return {"loss": tot / len(p), "modulating": mods, "gamma": g,
            "note": "an easy pixel with p_t = 0.9 keeps only "
                    "(1-0.9)^gamma of its weight"}


def dice_loss(prob, target):
    r""":math:`1 - 2|X\cap Y|/(|X|+|Y|)`, an overlap score."""
    p = [float(v) for v in k.vec(prob)]
    t = [float(v) for v in k.vec(target)]
    if len(p) != len(t):
        raise ValueError("samdec: the prediction and target differ "
                         "in size")
    inter = sum(p[i] * t[i] for i in range(len(p)))
    tot = sum(p) + sum(t)
    if tot <= _EPS:
        return {"loss": 0.0, "dice": 1.0,
                "note": "both empty, which is a perfect match"}
    d = 2.0 * inter / tot
    return {"loss": 1.0 - d, "dice": d}


def decode_mask(prompt_tokens, image_tokens, grid_shape,
                n_blocks=2, upsample_factor=2, output_index=0):
    r"""Two two-way blocks, upsample, dynamic head."""
    P = [[float(v) for v in r] for r in k.mat(prompt_tokens)]
    I = [[float(v) for v in r] for r in k.mat(image_tokens)]
    H, W = int(grid_shape[0]), int(grid_shape[1])
    if H * W != len(I):
        raise ValueError("samdec: %d image tokens do not fill a "
                         "%dx%d grid" % (len(I), H, W))
    for _ in range(int(n_blocks)):
        r = two_way_block(P, I)
        P, I = r["prompt_tokens"], r["image_tokens"]
    f = int(upsample_factor)
    grid = [[I[i * W + j] for j in range(W)] for i in range(H)]
    big = [[grid[i // f][j // f] for j in range(W * f)]
           for i in range(H * f)]
    head = dynamic_mask_head(P[int(output_index)], big)
    return RichResult(payload={
        "estimate": head["probability"], "mask": head["probability"],
        "logits": head["logits"], "shape": (H * f, W * f),
        "n_blocks": int(n_blocks),
        "method": "SAM mask decoder; Kirillov et al. (2023)",
        "note": "two-way attention updates prompt AND image, then a "
                "dynamic linear classifier built from the output "
                "token scores every location",
    })


def cheatsheet():
    return ("samdec: image embedding + prompt embeddings + a learned "
            "OUTPUT TOKEN -> mask. The decoder block does prompt "
            "self-attention and cross-attention in BOTH directions, so "
            "both embeddings are updated -- one direction would let "
            "the prompt read the image without the image knowing what "
            "was asked. After two blocks the image embedding is "
            "upsampled and an MLP turns the output token into a "
            "DYNAMIC linear classifier, so the mask is a dot product "
            "against weights built from the prompt. Loss is FOCAL + "
            "DICE, both chosen for the foreground/background "
            "imbalance.")


# compact alias per ledger/NAMING.md
sammaskdecoder = decode_mask
