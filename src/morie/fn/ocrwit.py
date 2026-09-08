# morie.fn -- function file (rootcoder007/morie)
r"""LayoutLMv3: one masking recipe for text and image alike.

Multimodal document models had pre-trained the image side with a
*different* objective from the text side -- masked language modelling
for words, and a region- or pixel-level objective for pictures. Two
objectives means two representation spaces and a bridge between them.

**The change is symmetry.** Text tokens are masked and reconstructed;
image patches are masked and reconstructed **as discrete tokens** from
a learned visual vocabulary. Both objectives are now "recover masked
units of a discrete sequence", so one encoder learns one joint space.

**No CNN backbone.** Patches are embedded linearly, as in a vision
Transformer, which removes the region proposals and object detector
earlier document models required, along with their parameters and
preprocessing.

**Word-patch alignment is what actually binds the modalities.** For an
*unmasked* text token, predict whether its corresponding image patch
was masked. Neither reconstruction objective requires knowing which
patch a word sits on; this one does. It is computed only for unmasked
words -- a masked word would leak its own reconstruction target.

**Layout enters as segment-level 2D positions**: a text line's box is
shared by its words, which is cheaper than per-word boxes and closer
to how documents are structured.

References
----------
Huang, Y., Lv, T., Cui, L., Lu, Y. & Wei, F. (2022) "LayoutLMv3:
Pre-training for Document AI with Unified Text and Image Masking",
*Proceedings of the 30th ACM International Conference on Multimedia
(MM '22)*, 4083-4091, doi:10.1145/3503161.3548112,
arXiv:2204.08387. That existing multimodal document models pre-train
the image modality with different objectives from the text modality;
the unification through masked language modelling and masked IMAGE
modelling with discrete image tokens; the word-patch alignment
objective predicting whether the corresponding image patch of a text
word is masked; and linear image patch embeddings in place of a CNN
backbone.

Dosovitskiy, A. et al. (2021) "An Image is Worth 16x16 Words",
*ICLR 2021*, arXiv:2010.11929. The linear patch embedding.

Bao, H., Dong, L., Piao, S. & Wei, F. (2022) "BEiT: BERT Pre-Training
of Image Transformers", *ICLR 2022*, arXiv:2106.08254. The discrete
visual tokens used as targets.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["normalise_bbox", "segment_layout_boxes", "mask_units",
           "patch_of_box", "word_patch_alignment"]

_EPS = 1e-12


def normalise_bbox(box, width, height, scale=1000):
    r"""Boxes onto a fixed integer grid, independent of page size."""
    x0, y0, x1, y1 = [float(v) for v in box]
    W, H = float(width), float(height)
    if W <= 0.0 or H <= 0.0:
        raise ValueError("ocrwit: the page dimensions must be "
                         "positive")
    if x1 < x0 or y1 < y0:
        raise ValueError("ocrwit: the box is inverted")
    s = int(scale)
    return [min(s, max(0, int(round(x0 / W * s)))),
            min(s, max(0, int(round(y0 / H * s)))),
            min(s, max(0, int(round(x1 / W * s)))),
            min(s, max(0, int(round(y1 / H * s))))]


def segment_layout_boxes(boxes, segment_ids, width, height,
                         scale=1000):
    r"""SEGMENT-level 2D position: a line's box shared by its words."""
    segs = list(segment_ids)
    B = list(boxes)
    if len(segs) != len(B):
        raise ValueError("ocrwit: %d boxes but %d segment ids"
                         % (len(B), len(segs)))
    by_seg = {}
    for i in range(len(segs)):
        by_seg.setdefault(segs[i], []).append(
            normalise_bbox(B[i], width, height, scale))
    seg_box = {}
    for s, bs in by_seg.items():
        seg_box[s] = [min(b[0] for b in bs), min(b[1] for b in bs),
                      max(b[2] for b in bs), max(b[3] for b in bs)]
    return {"segment_boxes": seg_box,
            "per_token": [seg_box[segs[i]] for i in range(len(segs))],
            "n_segments": len(seg_box),
            "note": "one box per segment, cheaper than per word and "
                    "closer to the document's structure"}


def mask_units(n_units, rate=0.3, seed=0, block=1):
    r"""Mask text tokens or image patches by the SAME procedure."""
    n = int(n_units)
    r = float(rate)
    if n < 1:
        raise ValueError("ocrwit: there is nothing to mask")
    if not 0.0 < r < 1.0:
        raise ValueError("ocrwit: the mask rate must lie in (0,1)")
    rng = np.random.default_rng(seed)
    b = max(1, int(block))
    masked = set()
    target = max(1, int(round(n * r)))
    guard = 0
    while len(masked) < target and guard < 1000 * n:
        s = int(float(rng.uniform()) * n) % n
        for j in range(s, min(n, s + b)):
            masked.add(j)
        guard += 1
    return {"masked": sorted(masked),
            "kept": sorted(set(range(n)) - masked),
            "rate": len(masked) / float(n), "block": b,
            "note": "the same recipe for both modalities, which is "
                    "the unification"}


def patch_of_box(box, width, height, patch_grid=14):
    r"""Which image patches a text box covers."""
    g = int(patch_grid)
    x0, y0, x1, y1 = normalise_bbox(box, width, height, g)
    out = []
    for r in range(min(y0, g - 1), min(max(y1, y0 + 1), g)):
        for c in range(min(x0, g - 1), min(max(x1, x0 + 1), g)):
            out.append(r * g + c)
    return sorted(set(out))


def word_patch_alignment(text_boxes, masked_patches, width, height,
                         patch_grid=14, masked_text=()):
    r"""For each UNMASKED word, is its patch masked?

    The objective that ties the modalities together, and the reason
    the model must know which patch a word sits on.
    """
    mp = set(int(v) for v in masked_patches)
    mt = set(int(v) for v in masked_text)
    labels, covered = {}, {}
    for i, b in enumerate(text_boxes):
        if i in mt:
            continue
        ps = patch_of_box(b, width, height, patch_grid)
        covered[i] = ps
        labels[i] = 1 if any(p in mp for p in ps) else 0
    if not labels:
        raise ValueError("ocrwit: every text token is masked, so the "
                         "alignment objective has no examples")
    return RichResult(payload={
        "estimate": labels, "labels": labels, "patches": covered,
        "n_examples": len(labels),
        "positive_rate": sum(labels.values()) / float(len(labels)),
        "method": "word-patch alignment; Huang, Lv, Cui, Lu & Wei "
                  "(2022)",
        "note": "unmasked words only -- a masked word would leak its "
                "own reconstruction target",
    })


def cheatsheet():
    return ("ocrwit: document models pre-trained text and image with "
            "DIFFERENT objectives, giving two spaces and a bridge. "
            "LayoutLMv3 makes them symmetric -- mask and reconstruct "
            "text tokens, mask and reconstruct image patches as "
            "DISCRETE tokens -- so one encoder learns one space. "
            "Linear patch embeddings, so no CNN backbone or detector. "
            "WORD-PATCH ALIGNMENT binds them: for an UNMASKED word, "
            "predict whether its patch was masked, which is the only "
            "objective that forces the model to know where a word "
            "sits. Layout is SEGMENT-level 2D position.")


# compact alias per ledger/NAMING.md
layoutlmv3 = word_patch_alignment

# public names resolved by fn/_lazy_map.json
ocr_wit_layout = word_patch_alignment
ocrwitlayout = word_patch_alignment
