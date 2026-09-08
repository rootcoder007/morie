# morie.fn -- function file (rootcoder007/morie)
r"""Segment Anything: promptable segmentation as a pre-training task.

The task is stated as a requirement rather than a metric: given any
prompt -- a point, a box, a rough mask -- return a **valid** mask, and
**when the prompt is ambiguous** (a point on a shirt may mean the
shirt or the person wearing it) return a reasonable mask for at least
one of the intended objects. That requirement is what makes the task
usable both as a pre-training objective and as a route to zero-shot
transfer by prompt engineering.

**The architecture follows from three constraints**: a flexible
prompt, amortised real-time computation, and ambiguity-awareness. A
heavy image encoder runs **once per image**; a light prompt encoder
and mask decoder run per prompt in about 50 ms. So the same image
embedding is reused across prompts and its cost is amortised --
``amortised_cost`` computes that, since the split is the entire reason
interactive use is possible.

**Prompts come in two kinds.** Sparse ones (points, boxes, text) are
positional encodings summed with a learned embedding **per prompt
type**, so a foreground click and a background click at the same place
are different objects to the model. Dense ones (masks) are embedded
with convolutions and **summed with the image embedding** rather than
concatenated.

**Ambiguity is answered with multiple outputs**, not with a
confidence interval; the ranking and the minimum-loss training that
go with it live in :mod:`sammkr`, and the decoder in :mod:`samdec`.

References
----------
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson,
L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P. &
Girshick, R. (2023) "Segment Anything", *Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV 2023)*, 4015-4026,
arXiv:2304.02643. Sec. 2 (the promptable segmentation task, and the
requirement that the output be a reasonable mask for at least one
object even when the prompt is ambiguous), Sec. 3 (the three
constraints -- flexible prompting, amortised real-time computation,
ambiguity-awareness; the image encoder run once per image; sparse
prompts as positional encodings summed with learned per-type
embeddings and dense mask prompts embedded by convolution and summed
with the image embedding; ~50 ms per prompt in a browser), and Sec. 5
(SA-1B: over 1B masks on 11M licensed, privacy-respecting images).

Dosovitskiy, A. et al. (2021) "An Image is Worth 16x16 Words",
*ICLR 2021*, arXiv:2010.11929. The ViT image encoder.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P. & Girshick, R. (2022)
"Masked Autoencoders Are Scalable Vision Learners", *CVPR 2022*,
16000-16009, arXiv:2111.06377. The MAE pre-training used for it.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["encode_point_prompt", "encode_box_prompt",
           "encode_mask_prompt", "amortised_cost",
           "promptable_segment"]

_EPS = 1e-12
_TYPES = ("foreground", "background", "box_tl", "box_br")


def _pos_enc(x, y, dim=8, scale=1.0):
    r"""Fourier positional encoding of a normalised coordinate."""
    out = []
    for j in range(int(dim) // 2):
        f = (2.0 ** j) * math.pi * float(scale)
        out.append(math.sin(f * float(x)))
        out.append(math.cos(f * float(y)))
    return out


def encode_point_prompt(points, labels, dim=8, type_embeddings=None):
    r"""Positional encoding PLUS a learned embedding per prompt type.

    A foreground and a background click at the same location must not
    be the same input, which is exactly what the type embedding
    supplies.
    """
    P = [(float(a), float(b)) for a, b in points]
    L = [int(v) for v in labels]
    if len(P) != len(L):
        raise ValueError("samseg: %d points but %d labels"
                         % (len(P), len(L)))
    if any(v not in (0, 1) for v in L):
        raise ValueError("samseg: a point label must be 1 "
                         "(foreground) or 0 (background)")
    te = type_embeddings or {}
    out = []
    for i in range(len(P)):
        e = _pos_enc(P[i][0], P[i][1], dim)
        name = "foreground" if L[i] == 1 else "background"
        t = te.get(name, [0.0] * len(e))
        if len(t) != len(e):
            raise ValueError("samseg: the type embedding has the "
                             "wrong width")
        out.append([e[a] + t[a] for a in range(len(e))])
    return {"tokens": out, "n_prompts": len(out), "sparse": True,
            "note": "a background click at the same place is a "
                    "DIFFERENT token, by the type embedding"}


def encode_box_prompt(box, dim=8, type_embeddings=None):
    r"""A box is two corner points with their own type embeddings."""
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 <= x0 or y1 <= y0:
        raise ValueError("samseg: the box is empty or inverted")
    te = type_embeddings or {}
    a = _pos_enc(x0, y0, dim)
    b = _pos_enc(x1, y1, dim)
    ta = te.get("box_tl", [0.0] * len(a))
    tb = te.get("box_br", [0.0] * len(b))
    return {"tokens": [[a[i] + ta[i] for i in range(len(a))],
                       [b[i] + tb[i] for i in range(len(b))]],
            "n_prompts": 2, "sparse": True}


def encode_mask_prompt(mask, image_embedding, weight=1.0):
    r"""A dense prompt is SUMMED with the image embedding.

    Not concatenated: the mask lives on the same spatial grid, so
    adding it keeps the decoder's input shape independent of whether a
    mask prompt was given.
    """
    M = [[float(v) for v in r] for r in k.mat(mask)]
    E = [[float(v) for v in r] for r in k.mat(image_embedding)]
    if len(M) != len(E) or len(M[0]) != len(E[0]):
        raise ValueError("samseg: the mask prompt is %dx%d but the "
                         "image embedding is %dx%d"
                         % (len(M), len(M[0]), len(E), len(E[0])))
    w = float(weight)
    return {"embedding": [[E[i][j] + w * M[i][j]
                           for j in range(len(E[0]))]
                          for i in range(len(E))],
            "sparse": False,
            "note": "summed, so the decoder input shape is unchanged"}


def amortised_cost(encoder_ms, decoder_ms, n_prompts):
    r"""Why the split into a heavy encoder and a light decoder.

    The encoder runs ONCE per image; each further prompt costs only
    the decoder, so the per-prompt cost falls like :math:`1/P`.
    """
    e, d = float(encoder_ms), float(decoder_ms)
    P = int(n_prompts)
    if P < 1:
        raise ValueError("samseg: at least one prompt is needed")
    if e <= 0.0 or d <= 0.0:
        raise ValueError("samseg: the timings must be positive")
    total = e + P * d
    return {"total_ms": total, "per_prompt_ms": total / P,
            "naive_ms": P * (e + d),
            "speedup": P * (e + d) / total,
            "interactive": d < 100.0,
            "note": "the image embedding is computed once and reused"}


def promptable_segment(image_embedding, prompt_tokens, decoder,
                       multimask=True):
    r"""Run the light decoder against a cached image embedding.

    ``decoder`` maps (embedding, tokens, multimask) to a list of
    masks. With ``multimask=False`` a single output must AVERAGE the
    valid interpretations of an ambiguous prompt, which is the failure
    :mod:`sammkr` exists to avoid.
    """
    masks = decoder(image_embedding, prompt_tokens, multimask)
    if not masks:
        raise ValueError("samseg: the decoder returned no mask; the "
                         "task requires a valid mask for ANY prompt")
    return RichResult(payload={
        "estimate": masks[0], "masks": masks, "n_masks": len(masks),
        "multimask": bool(multimask),
        "method": "promptable segmentation; Kirillov et al. (2023)",
        "note": "a valid mask for any prompt, and for an ambiguous "
                "prompt a valid mask for at least one intended object",
    })


def cheatsheet():
    return ("samseg: the task is 'return a VALID mask for any prompt, "
            "and for an AMBIGUOUS prompt a valid mask for at least one "
            "intended object' -- which is what makes it usable as "
            "pre-training and for zero-shot transfer by prompting. "
            "Three constraints force the architecture: flexible "
            "prompts, amortised real-time use, ambiguity-awareness. "
            "So a heavy image encoder runs ONCE per image and a light "
            "prompt encoder plus mask decoder run per prompt (~50 ms). "
            "Sparse prompts are positional encodings plus a learned "
            "PER-TYPE embedding; dense mask prompts are SUMMED with "
            "the image embedding.")


# compact alias per ledger/NAMING.md
segmentanything = promptable_segment

# public names resolved by fn/_lazy_map.json
sam_segment = promptable_segment
samsegment = promptable_segment
