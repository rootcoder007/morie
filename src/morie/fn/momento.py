# morie.fn -- function file (rootcoder007/morie)
r"""MOMENT: masked time-series modelling across many datasets.

Masked language modelling learns from unlabelled text by hiding tokens
and reconstructing them; masked image modelling does the same for
patches. MOMENT applies that recipe to time series -- mask portions of
the input and train the model to reconstruct them -- and the interest
is in why this had not simply been done already.

**Time series do not come in a common format.** Text and images are
largely uniform: consistent sampling, a fixed number of channels.
Series vary in temporal resolution, in channel count, in length, in
amplitude, and they have missing values. That is why almost everything
before was trained on one dataset and transferred, with modest
success, rather than pretrained across many. Making mixed-dataset
pretraining work is the contribution, and it forces the preprocessing
to be explicit: a common patch length, per-series normalisation, and
channel independence so a 3-channel dataset and a 300-channel dataset
can sit in the same batch.

**Masking replaces patches with zeros.** Following the established
practice this module implements, masked patches are zeroed and the
model reconstructs them, with the loss computed **only on the masked
positions** -- reconstructing what was already visible teaches
nothing. That restriction is the part worth getting right, and the
anchor checks it directly.

**Why the mask rate is a real knob.** Mask too little and the task is
trivially solved by interpolation from immediate neighbours. Mask too
much and there is not enough context to reconstruct anything, so the
model learns the dataset mean. Neither extreme produces a useful
representation, and both are visible in the reconstruction error as a
function of mask rate rather than being matters of opinion.

**One pretrained model, several tasks.** Because the objective is
reconstruction rather than forecasting, the same encoder serves
forecasting, classification, anomaly detection and imputation -- the
masked span is simply placed differently. Imputation masks an interior
gap; forecasting masks the tail. ``mask_patches`` takes the span
explicitly for that reason.

References
----------
Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S. & Dubrawski,
A. (2024) "MOMENT: A Family of Open Time-series Foundation Models",
*Proceedings of the 41st International Conference on Machine
Learning*, PMLR 235, arXiv:2402.03885. The masked time-series
prediction objective; contribution C2 on multi-dataset pretraining
and why varying resolution, channel count, length and amplitude made
it largely unexplored; the practice of masking with zeros and
reconstructing; and contribution C3 on multi-task evaluation.

Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019) "BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding", *NAACL-HLT 2019*, arXiv:1810.04805. The masked
modelling objective being transferred.

Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. (2023) "A Time
Series is Worth 64 Words: Long-term Forecasting with Transformers",
*ICLR 2023*, arXiv:2211.14730. The patching and channel-independence
conventions this pretraining rests on.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["harmonise", "mask_patches", "masked_loss",
           "reconstruction_curve", "task_mask"]

_EPS = 1e-12
_TASKS = ("forecast", "impute", "classify", "anomaly")


def harmonise(series_list, patch_len, normalise=True):
    r"""Put series of differing length, scale and channel count into
    one batch.

    Each series is normalised on its own and truncated to a whole
    number of patches. Channels are kept independent, so datasets with
    different channel counts coexist without padding one to the other.
    """
    P = int(patch_len)
    if P < 1:
        raise ValueError("momento: patch_len must be at least 1")
    if not series_list:
        raise ValueError("momento: no series given")
    out, meta = [], []
    for s in series_list:
        M = [[float(v) for v in r] for r in k.mat(s)]
        if not M:
            raise ValueError("momento: one of the series is empty")
        D = len(M[0])
        L = (len(M) // P) * P
        if L < P:
            raise ValueError("momento: a series has %d points, fewer "
                             "than one patch of %d" % (len(M), P))
        for d in range(D):
            col = [M[t][d] for t in range(L)]
            if normalise:
                m = sum(col) / L
                sd = math.sqrt(sum((v - m) ** 2 for v in col)
                               / max(L - 1, 1))
                col = ([0.0] * L if sd <= _EPS
                       else [(v - m) / sd for v in col])
            else:
                m, sd = 0.0, 1.0
            out.append([col[i * P:(i + 1) * P] for i in range(L // P)])
            meta.append({"mean": m, "sd": sd, "n_patches": L // P})
    n = min(x["n_patches"] for x in meta)
    return {"batch": [row[:n] for row in out], "meta": meta,
            "n_series": len(out), "n_patches": n, "patch_len": P,
            "note": "each channel is its own row, so datasets with "
                    "different channel counts share a batch"}


def mask_patches(patches, mask_idx, fill=0.0):
    r"""Replace the named patches with zeros, as the practice is.

    Returns the masked input and the boolean mask, because the loss
    must be computed on the masked positions only.
    """
    P = [[float(v) for v in p] for p in patches]
    n = len(P)
    idx = sorted(set(int(i) for i in mask_idx))
    if any(not 0 <= i < n for i in idx):
        raise ValueError("momento: a mask index is outside 0..%d"
                         % (n - 1))
    if not idx:
        raise ValueError("momento: nothing was masked, so there is "
                         "nothing to learn from")
    if len(idx) == n:
        raise ValueError("momento: every patch was masked, leaving no "
                         "context to reconstruct from")
    masked = [([float(fill)] * len(P[i]) if i in idx else list(P[i]))
              for i in range(n)]
    return {"masked": masked, "mask": [i in idx for i in range(n)],
            "mask_idx": idx, "mask_rate": len(idx) / float(n),
            "n_patches": n}


def masked_loss(truth, reconstruction, mask):
    r"""Mean squared error **on the masked positions only**.

    Scoring the visible positions would reward copying the input, so
    they are excluded. The count of scored positions is returned so a
    silently empty loss is impossible.
    """
    T = [[float(v) for v in p] for p in truth]
    R = [[float(v) for v in p] for p in reconstruction]
    if len(T) != len(R) or len(T) != len(mask):
        raise ValueError("momento: truth, reconstruction and mask "
                         "must agree in length (%d, %d, %d)"
                         % (len(T), len(R), len(mask)))
    tot, cnt = 0.0, 0
    for i in range(len(T)):
        if not mask[i]:
            continue
        if len(T[i]) != len(R[i]):
            raise ValueError("momento: patch %d differs in length "
                             "between truth and reconstruction" % i)
        for j in range(len(T[i])):
            tot += (T[i][j] - R[i][j]) ** 2
            cnt += 1
    if cnt == 0:
        raise ValueError("momento: no position was masked, so the "
                         "loss is undefined")
    return {"mse": tot / cnt, "n_scored": cnt,
            "scored": "masked positions only -- scoring the visible "
                      "ones would reward copying"}


def task_mask(n_patches, task="forecast", span=1, start=None):
    r"""Where to put the mask for each downstream task.

    Forecasting masks the tail; imputation masks an interior gap. The
    objective does not change -- only the placement -- which is what
    lets one pretrained encoder serve several tasks.
    """
    n = int(n_patches)
    s = int(span)
    if task not in _TASKS:
        raise ValueError("momento: task must be one of %s, got %r"
                         % (", ".join(_TASKS), task))
    if not 1 <= s < n:
        raise ValueError("momento: the span must lie in 1..%d, got %d"
                         % (n - 1, s))
    if task == "forecast":
        return list(range(n - s, n))
    if task == "impute":
        st = int(start) if start is not None else max(1, (n - s) // 2)
        if st + s > n:
            raise ValueError("momento: the imputation gap runs past "
                             "the end")
        return list(range(st, st + s))
    return list(range(n - s, n))


def reconstruction_curve(patches, reconstructor, rates, seed=0):
    r"""Reconstruction error against mask rate.

    Both extremes are failure modes: too little masking makes the task
    solvable by interpolation, too much removes the context needed to
    solve it at all. The curve shows them rather than asserting them.
    """
    P = [[float(v) for v in p] for p in patches]
    n = len(P)
    rng = np.random.default_rng(seed)
    out = []
    for r in rates:
        m = max(1, min(n - 1, int(round(float(r) * n))))
        idx = sorted(range(n),
                     key=lambda _i: float(rng.uniform()))[:m]
        mk = mask_patches(P, idx)
        rec = reconstructor(mk["masked"], mk["mask"])
        L = masked_loss(P, rec, mk["mask"])
        out.append({"rate": mk["mask_rate"], "mse": L["mse"],
                    "n_masked": m})
    return {"curve": out, "n_patches": n,
            "rates": [o["rate"] for o in out],
            "mse": [o["mse"] for o in out]}


def cheatsheet():
    return ("momento: masked time-series pretraining. Mask patches "
            "with ZEROS and reconstruct; the loss counts the MASKED "
            "positions only, since scoring visible ones rewards "
            "copying. The hard part is multi-dataset pretraining: "
            "series differ in resolution, channel count, length and "
            "amplitude, so harmonise per-series and keep channels "
            "independent. Mask rate is a real knob -- too low is "
            "interpolation, too high leaves no context. Task changes "
            "only WHERE the mask goes: tail for forecasting, interior "
            "for imputation.")


# compact alias per ledger/NAMING.md
momentfoundation = harmonise

# public names resolved by fn/_lazy_map.json
moment_foundation = harmonise
