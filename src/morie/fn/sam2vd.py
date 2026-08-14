# morie.fn -- function file (rootcoder007/morie)
r"""SAM 2: the same model, plus a memory of what it already segmented.

An image is a static snapshot; a video is the same objects deforming,
occluding one another, leaving the frame and coming back. SAM 2
generalises promptable segmentation to video by keeping a **streaming
memory**: frames arrive one at a time, and the current frame's
features are conditioned on memories of past frames before the mask
decoder ever sees them.

**The reduction to SAM is exact and is the design claim.** With an
empty memory bank the memory attention has nothing to attend to and
the model behaves exactly like SAM on a single image. ``propagate``
therefore returns the unconditioned features unchanged on the first
frame, and the anchor asserts that equality rather than a tolerance.

**The memory bank is two FIFO queues, not one.** Up to :math:`N`
recent frames, and separately up to :math:`M` **prompted** frames. In
the common video-object-segmentation case the only prompt is the first
frame's mask, so the bank must keep that memory permanently while
recent memories churn past it -- one queue would evict the thing the
user actually specified.

**Temporal position embeddings go on the recent memories only.** The
recent frames carry short-term motion, so their ordering is
informative. Prompted frames are deliberately left unembedded: the
training signal from them is sparser, and at inference they may come
from a temporal range never seen in training, so encoding their
distance would not generalise.

**Object pointers are separate from spatial memory.** Lightweight
vectors taken from the mask decoder's output tokens carry high-level
semantics ("this object"), and memory attention cross-attends to both
them and the spatial feature maps -- appearance can change completely
while the pointer stays the same object.

References
----------
Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H.,
Radle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala,
K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollar, P. &
Feichtenhofer, C. (2024) "SAM 2: Segment Anything in Images and
Videos", arXiv:2408.00714. Sec. 4: the streaming architecture
processing video frames one at a time with a memory attention module
attending to previous memories of the target object, and that when
applied to images the memory is empty and the model behaves like SAM;
the memory attention stacking L blocks of self-attention followed by
cross-attention to memories of prompted and unprompted frames and to
OBJECT POINTERS, followed by an MLP; the memory bank as a FIFO queue
of up to N recent frames plus a separate FIFO queue of up to M
prompted frames, both stored as spatial feature maps; the object
pointers taken from mask decoder output tokens; and temporal position
information embedded into the N recent memories but NOT into prompted
frames, because their training signal is sparser and generalising to
unseen temporal ranges is harder. Also the reported results: 3x fewer
interactions in video and 6x faster than SAM on images, with the SA-V
dataset of 35.5M masks across 50.9K videos.

Kirillov, A. et al. (2023) "Segment Anything", *ICCV 2023*,
4015-4026, arXiv:2304.02643. The image model generalised here;
implemented in :mod:`samseg` and :mod:`samdec`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["memory_bank", "push_memory", "temporal_embedding",
           "memory_attention", "propagate"]

_EPS = 1e-12


def memory_bank(n_recent=7, m_prompted=1):
    r"""Two FIFO queues: recent frames, and PROMPTED frames."""
    N, M = int(n_recent), int(m_prompted)
    if N < 1 or M < 1:
        raise ValueError("sam2vd: both queue capacities must be >= 1")
    return {"recent": [], "prompted": [], "pointers": [],
            "n_recent": N, "m_prompted": M,
            "note": "one queue would evict the frame the user "
                    "actually prompted"}


def push_memory(bank, frame_index, features, prompted=False,
                object_pointer=None):
    r"""Add a memory, evicting only within its own queue."""
    b = dict(bank)
    b["recent"] = list(bank["recent"])
    b["prompted"] = list(bank["prompted"])
    b["pointers"] = list(bank["pointers"])
    entry = {"frame": int(frame_index),
             "features": [float(v) for v in k.vec(features)],
             "prompted": bool(prompted)}
    if prompted:
        b["prompted"].append(entry)
        if len(b["prompted"]) > b["m_prompted"]:
            b["prompted"].pop(0)
    else:
        b["recent"].append(entry)
        if len(b["recent"]) > b["n_recent"]:
            b["recent"].pop(0)
    if object_pointer is not None:
        b["pointers"].append({
            "frame": int(frame_index),
            "vector": [float(v) for v in k.vec(object_pointer)]})
        cap = b["n_recent"] + b["m_prompted"]
        if len(b["pointers"]) > cap:
            b["pointers"].pop(0)
    return b


def temporal_embedding(entry, current_frame, dim=None, scale=0.1):
    r"""Position in time -- for RECENT memories only.

    Prompted frames are left unembedded on purpose: at inference they
    may sit at a temporal distance never seen during training.
    """
    v = list(entry["features"])
    if entry["prompted"]:
        return {"features": v, "embedded": False,
                "note": "prompted memories carry no temporal "
                        "position, by design"}
    d = int(current_frame) - int(entry["frame"])
    n = len(v) if dim is None else int(dim)
    out = list(v)
    for i in range(min(n, len(v))):
        out[i] = v[i] + math.sin(float(scale) * d * (i + 1))
    return {"features": out, "embedded": True, "distance": d}


def memory_attention(frame_features, bank, current_frame,
                     n_blocks=1, include_pointers=True):
    r"""Condition the current frame on the memory bank.

    With an empty bank this is the identity, which is precisely the
    claim that SAM 2 reduces to SAM on a single image.
    """
    x = [float(v) for v in k.vec(frame_features)]
    mem = []
    for e in bank["prompted"] + bank["recent"]:
        t = temporal_embedding(e, current_frame)
        if len(t["features"]) != len(x):
            raise ValueError("sam2vd: a memory has width %d but the "
                             "frame has %d"
                             % (len(t["features"]), len(x)))
        mem.append(t["features"])
    if include_pointers:
        for p in bank["pointers"]:
            if len(p["vector"]) == len(x):
                mem.append(p["vector"])
    if not mem:
        return {"features": x, "attended": False, "n_memories": 0,
                "weights": [],
                "note": "empty memory: the model IS SAM here"}
    out = list(x)
    for _ in range(int(n_blocks)):
        d = len(out)
        sc = [sum(out[a] * m[a] for a in range(d)) / math.sqrt(d)
              for m in mem]
        top = max(sc)
        e = [math.exp(v - top) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        ctx = [sum(w[j] * mem[j][a] for j in range(len(mem)))
               for a in range(d)]
        out = [out[a] + ctx[a] for a in range(d)]
    return {"features": out, "attended": True, "n_memories": len(mem),
            "weights": w,
            "note": "self-attention, then cross-attention to spatial "
                    "memories AND object pointers"}


def propagate(frames, encoder, decoder, prompts=None, n_recent=7,
              m_prompted=1):
    r"""Stream through the video, one frame at a time.

    ``prompts`` maps a frame index to a prompt; a prompted frame's
    memory goes in the prompted queue and is never evicted by recent
    traffic.
    """
    P = dict(prompts or {})
    bank = memory_bank(n_recent, m_prompted)
    masks, conditioned = [], []
    for t, f in enumerate(frames):
        raw = [float(v) for v in k.vec(encoder(f))]
        att = memory_attention(raw, bank, t)
        conditioned.append(att["attended"])
        m = decoder(att["features"], P.get(t))
        masks.append(m)
        bank = push_memory(bank, t, att["features"],
                           prompted=(t in P),
                           object_pointer=att["features"])
    return RichResult(payload={
        "estimate": masks, "masks": masks,
        "conditioned": conditioned, "n_frames": len(masks),
        "first_frame_is_sam": conditioned[0] is False
        if conditioned else True,
        "method": "streaming memory propagation; Ravi et al. (2024)",
        "note": "frame 0 has an empty memory, so it is exactly the "
                "image model",
    })


def cheatsheet():
    return ("sam2vd: video is the same objects deforming, occluding "
            "and re-appearing, so carry a STREAMING MEMORY -- condition "
            "each frame's features on memories of past frames before "
            "decoding. With an EMPTY memory the model is exactly SAM, "
            "which is the design claim. The bank is TWO FIFO queues: N "
            "recent frames and M PROMPTED frames, because one queue "
            "would evict the frame the user specified. Temporal "
            "position embeddings go on recent memories only -- prompted "
            "frames may sit at distances never trained on. OBJECT "
            "POINTERS carry identity when appearance changes "
            "completely.")


# compact alias per ledger/NAMING.md
sam2video = propagate
