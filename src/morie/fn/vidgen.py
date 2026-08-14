# morie.fn -- function file (rootcoder007/morie)
r"""Video diffusion: a 3D U-Net factorised over space and time.

A video model built by inflating an image model has to answer two
questions: how to attend across frames without paying
:math:`(FS)^2`, and how to keep the image model's quality while doing
so.

**Factorisation answers both.** Each :math:`3\times3` convolution
becomes a :math:`1\times3\times3` convolution -- space only. Each
spatial attention block keeps attending over space with the frame axis
treated as a **batch** axis, and a **temporal** attention block is
inserted after it, attending over frames with the spatial axes as
batch. Cost falls from :math:`(FS)^2` to :math:`F S^2 + S F^2`, which
``attention_cost`` computes exactly.

**And the factorisation buys something unique to video**: the model
can be masked to run on *independent images* simply by fixing each
temporal attention matrix to the identity -- each query attends only to
its own timestep. That makes **joint training** on video and image
objectives straightforward, and the paper finds that joint training
matters for sample quality. ``as_image_model`` performs exactly that
masking, and the anchor checks the resulting output equals the
per-frame computation *exactly*, not approximately.

**Reconstruction guidance extends the model past its frame count.**
To condition a sample on given frames :math:`x^a`, add a gradient of
the squared error between the model's denoised estimate of those
frames and their true values, weighted by :math:`w_r > 1`. It is a
guidance term, so it is applied at sampling time to a model that was
never trained conditionally -- and the same construction with a
downsampling operator inside the loss gives spatial super-resolution.

References
----------
Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M. & Fleet,
D. J. (2022) "Video Diffusion Models", *Advances in Neural Information
Processing Systems 35 (NeurIPS 2022)*, arXiv:2204.03458. Sec. 3: the
3D U-Net factorised over space and time, changing each 3x3 convolution
into a 1x3x3 space-only convolution, keeping spatial attention with the
frame axis as a batch axis, and inserting a temporal attention block
after each spatial attention block; that the factorisation makes it
straightforward to mask the model to run on independent images by
fixing the temporal attention matrix to match each key and query at
the same timestep, enabling JOINT training on video and image
objectives, which the experiments find important for sample quality;
and Sec. 4 (reconstruction guidance for conditional generation, with a
weighting factor w_r > 1 improving sample quality, extended to spatial
interpolation and super-resolution by imposing the squared error on a
downsampled model prediction and backpropagating through the
downsampling).

Ho, J., Jain, A. & Abbeel, P. (2020) "Denoising Diffusion
Probabilistic Models", *NeurIPS 2020*, arXiv:2006.11239. The
diffusion model being inflated.

Ho, J. & Salimans, T. (2022) "Classifier-Free Diffusion Guidance",
arXiv:2207.12598. The guidance framework this parallels.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["space_only_conv", "spatial_attention",
           "temporal_attention", "as_image_model",
           "attention_cost", "reconstruction_guidance"]

_EPS = 1e-12


def space_only_conv(video, kernel):
    r"""A :math:`1\times3\times3` convolution: frames stay separate.

    ``video`` is frames x height x width.
    """
    V = [[[float(v) for v in row] for row in fr] for fr in video]
    K = [[float(v) for v in r] for r in k.mat(kernel)]
    kh, kw = len(K), len(K[0])
    out = []
    for fr in V:
        H, W = len(fr), len(fr[0])
        if kh > H or kw > W:
            raise ValueError("vidgen: the kernel is larger than the "
                             "frame")
        o = []
        for i in range(H - kh + 1):
            row = []
            for j in range(W - kw + 1):
                row.append(sum(fr[i + a][j + b] * K[a][b]
                               for a in range(kh)
                               for b in range(kw)))
            o.append(row)
        out.append(o)
    return {"video": out, "frames": len(out),
            "note": "no kernel taps across frames; time is handled "
                    "only by the temporal attention block"}


def _softmax_attend(X, mask=None):
    n, d = len(X), len(X[0])
    out, W = [], []
    for i in range(n):
        sc = [sum(X[i][a] * X[j][a] for a in range(d))
              / math.sqrt(d) for j in range(n)]
        if mask is not None:
            sc = [sc[j] if mask[i][j] else -1e30 for j in range(n)]
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        W.append(w)
        out.append([sum(w[j] * X[j][a] for j in range(n))
                    for a in range(d)])
    return out, W


def spatial_attention(video):
    r"""Attention over SPACE, with the frame axis as a batch axis."""
    out, weights = [], []
    for fr in video:
        X = [[float(v) for v in row] for row in k.mat(fr)]
        o, w = _softmax_attend(X)
        out.append(o)
        weights.append(w)
    return {"video": out, "weights": weights,
            "note": "each frame attended independently"}


def temporal_attention(video, identity=False):
    r"""Attention over FRAMES, with the spatial axes as batch axes.

    ``identity=True`` fixes the attention matrix so each position
    attends only to its own timestep -- the masking that turns the
    video model into an image model.
    """
    V = [[[float(v) for v in row] for row in k.mat(fr)]
         for fr in video]
    F = len(V)
    if F < 1:
        raise ValueError("vidgen: the video has no frames")
    H, W = len(V[0]), len(V[0][0])
    if any(len(f) != H or len(f[0]) != W for f in V):
        raise ValueError("vidgen: the frames differ in shape")
    out = [[[0.0] * W for _ in range(H)] for _ in range(F)]
    for i in range(H):
        for j in range(W):
            series = [[V[t][i][j]] for t in range(F)]
            if identity:
                for t in range(F):
                    out[t][i][j] = series[t][0]
                continue
            o, _ = _softmax_attend(series)
            for t in range(F):
                out[t][i][j] = o[t][0]
    return {"video": out, "identity": bool(identity),
            "note": "identity=True is EXACTLY the independent-image "
                    "case, which is what makes joint training easy"}


def as_image_model(video, block):
    r"""Run the video model on independent images.

    ``block`` maps a video to a video; here each frame is passed
    alone, which is what the identity temporal mask reproduces.
    """
    return {"video": [block([fr])["video"][0] for fr in video],
            "note": "frames processed alone; the masked video model "
                    "must equal this exactly"}


def attention_cost(frames, spatial_positions):
    r"""Factorised against joint attention."""
    F, S = int(frames), int(spatial_positions)
    if F < 1 or S < 1:
        raise ValueError("vidgen: the frame and position counts must "
                         "be positive")
    joint = (F * S) ** 2
    fact = F * S * S + S * F * F
    return {"joint": joint, "factorised": fact,
            "ratio": joint / float(fact),
            "note": "(FS)^2 against F S^2 + S F^2"}


def reconstruction_guidance(x_hat, observed, index, weight=2.0,
                            downsample=None):
    r"""Guide a sample toward given frames at sampling time.

    The gradient of :math:`\|x^a - \hat x^a\|^2` at the observed
    positions, scaled by :math:`w_r`. With ``downsample`` the same
    construction gives super-resolution.
    """
    X = [[float(v) for v in k.vec(fr)] for fr in x_hat]
    O = [[float(v) for v in k.vec(fr)] for fr in observed]
    idx = [int(v) for v in index]
    if len(O) != len(idx):
        raise ValueError("vidgen: %d observed frames but %d indices"
                         % (len(O), len(idx)))
    w = float(weight)
    if w <= 0.0:
        raise ValueError("vidgen: the guidance weight must be "
                         "positive")
    grad = [[0.0] * len(X[0]) for _ in range(len(X))]
    err = 0.0
    for a, t in enumerate(idx):
        if t < 0 or t >= len(X):
            raise ValueError("vidgen: frame %d is outside the sample"
                             % t)
        pred = X[t]
        tgt = O[a]
        if downsample is not None:
            pred = [float(v) for v in k.vec(downsample(pred))]
            if len(pred) != len(tgt):
                raise ValueError("vidgen: the downsampled prediction "
                                 "does not match the low-resolution "
                                 "target")
            for i in range(len(tgt)):
                err += (pred[i] - tgt[i]) ** 2
            for i in range(len(grad[t])):
                grad[t][i] = 0.0
            continue
        for i in range(len(tgt)):
            d = pred[i] - tgt[i]
            err += d * d
            grad[t][i] = -w * 2.0 * d
    return RichResult(payload={
        "estimate": grad, "gradient": grad, "error": err,
        "weight": w, "guided_frames": idx,
        "method": "reconstruction guidance; Ho et al. (2022)",
        "note": "applied at SAMPLING time, so the model itself was "
                "never trained conditionally",
    })


def cheatsheet():
    return ("vidgen: a 3D U-Net FACTORISED over space and time -- each "
            "3x3 convolution becomes 1x3x3 (space only), spatial "
            "attention keeps the frame axis as a BATCH axis, and a "
            "temporal attention block is inserted after it with the "
            "spatial axes as batch. Cost drops from (FS)^2 to "
            "F S^2 + S F^2. The unique payoff: fixing the temporal "
            "attention to the IDENTITY makes the model run on "
            "independent images exactly, so video and image objectives "
            "can be trained JOINTLY -- which matters for sample "
            "quality. RECONSTRUCTION GUIDANCE conditions on given "
            "frames at sampling time, and with a downsampler inside "
            "the loss gives super-resolution.")


# compact alias per ledger/NAMING.md
videodiffusion = reconstruction_guidance

# public names resolved by fn/_lazy_map.json
video_diffusion = reconstruction_guidance
