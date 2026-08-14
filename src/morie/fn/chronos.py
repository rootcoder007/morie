# morie.fn -- function file (rootcoder007/morie)
r"""Chronos: forecasting by treating a series as a language.

The premise is deliberately unambitious: take an off-the-shelf language
model, change **nothing** about the architecture, and feed it time
series turned into tokens. If that works, the time-series-specific
machinery everyone builds may not have been necessary.

**Scaling, and why mean scaling specifically.** Series differ wildly in
scale even inside one dataset, which makes optimisation hard, so each
series is normalised by an affine map
:math:`\tilde x_i = (x_i - m)/s`. Chronos sets :math:`m = 0` and

.. math:: s = \frac{1}{C}\sum_{i=1}^{C} |x_i|,

the mean absolute value over the historical context. The choice of
:math:`m = 0` is not incidental: it means **zero maps to zero**, and
zeros in a series are usually semantically real -- no sales that day,
no solar generation at night. Standard scaling would move them.

**Quantisation into a fixed vocabulary.** Pick :math:`B` bin centres
:math:`c_1 < \dots < c_B` and :math:`B-1` edges with
:math:`c_i < b_i < c_{i+1}`. Under **uniform** binning the centres are
evenly spaced and each edge sits exactly midway,
:math:`b_i = (c_i + c_{i+1})/2`. Quantile binning -- bins carrying
equal numbers of training points -- is the alternative, and Chronos
rejects it: the value distribution of an unseen downstream dataset can
look nothing like the training one, so bins fitted to the training CDF
would be wrong in the wrong places.

**The limitation this creates, stated plainly.** Predictions can only
fall in :math:`[c_1, c_B]`. A series with a strong trend eventually
leaves that interval and becomes, in the paper's words, theoretically
infeasible to model. That is a real ceiling, not a tuning issue, and
``quantize`` reports how much of a series was clipped rather than
silently flattening it.

**What is thrown away on purpose.** No day-of-week, no week-of-year,
no frequency features -- the series is treated as a bare sequence.
Counter-intuitive, and the point: if calendar features were load-
bearing, the language-model framing would not work at all.

**Cross-entropy, not a distance.** Training uses the ordinary
categorical loss over the vocabulary, which knows nothing about bin
*order* -- predicting bin 5 when the answer is 6 costs the same as
predicting bin 500. The model has to learn that neighbouring bins are
similar from data alone. Forecasts are therefore distributions over
bins, and any summary -- mean, median, quantile -- is computed from
that distribution rather than being what the model emits.

References
----------
Ansari, A. F. et al. (2024) "Chronos: Learning the Language of Time
Series", *Transactions on Machine Learning Research* (10/2024),
arXiv:2403.07815. Sec. 3.1: mean scaling with m = 0 and
s = (1/C) sum |x_i| and its preservation of zeros; quantisation into
B bins with edges midway between centres; the choice of uniform over
quantile binning because downstream distributions differ; the
resulting restriction of predictions to [c_1, c_B] and its
consequence for trending series; the PAD and EOS tokens; and the
decision to ignore time and frequency information entirely.

Salinas, D., Flunkert, V., Gasthaus, J. & Januschowski, T. (2020)
"DeepAR: Probabilistic forecasting with autoregressive recurrent
networks", *International Journal of Forecasting* 36(3), 1181-1191,
doi:10.1016/j.ijforecast.2019.07.001. The mean-scaling scheme adopted
here.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena,
M., Zhou, Y., Li, W. & Liu, P. J. (2020) "Exploring the Limits of
Transfer Learning with a Unified Text-to-Text Transformer", *Journal
of Machine Learning Research* 21(140), 1-67, arXiv:1910.10683. The T5
family Chronos is built on, used unmodified apart from the vocabulary
size.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["mean_scale", "uniform_bins", "quantile_bins", "quantize",
           "dequantize", "tokenize", "detokenize", "forecast_summary"]

_EPS = 1e-12
PAD, EOS = -1, -2


def mean_scale(x, context=None):
    r"""Scale by the mean absolute value of the context, keeping zeros.

    :math:`m = 0`, :math:`s = \frac1C\sum|x_i|`. Because the shift is
    zero, an input of zero scales to zero exactly -- which is the
    reason this scheme is chosen over standard scaling.
    """
    v = [float(q) for q in k.vec(x)]
    if not v:
        raise ValueError("chronos: the series is empty")
    C = len(v) if context is None else int(context)
    if C < 1 or C > len(v):
        raise ValueError("chronos: the context length must lie in "
                         "1..%d, got %d" % (len(v), C))
    s = sum(abs(q) for q in v[:C]) / C
    if s <= _EPS:
        return {"scaled": [0.0] * len(v), "scale": 0.0,
                "degenerate": True,
                "note": "the context is all zeros, so no scale is "
                        "defined"}
    return {"scaled": [q / s for q in v], "scale": s,
            "degenerate": False, "context": C,
            "preserves_zero": True}


def uniform_bins(lo=-15.0, hi=15.0, n_bins=4096):
    r"""Evenly spaced centres with edges exactly midway between them."""
    B = int(n_bins)
    if B < 2:
        raise ValueError("chronos: need at least 2 bins, got %d" % B)
    if float(hi) <= float(lo):
        raise ValueError("chronos: hi must exceed lo")
    centers = [float(lo) + (float(hi) - float(lo)) * i / (B - 1)
               for i in range(B)]
    edges = [0.5 * (centers[i] + centers[i + 1]) for i in range(B - 1)]
    return {"centers": centers, "edges": edges, "n_bins": B,
            "scheme": "uniform",
            "range": (centers[0], centers[-1])}


def quantile_bins(samples, n_bins=4096):
    r"""Bins carrying roughly equal numbers of training points.

    Offered because the paper names it as the alternative, and
    rejected there as a default: an unseen dataset's values may sit
    where the training CDF put no bins at all.
    """
    v = sorted(float(q) for q in k.vec(samples))
    B = int(n_bins)
    if len(v) < B:
        raise ValueError("chronos: %d samples cannot define %d "
                         "quantile bins" % (len(v), B))
    centers = [v[min(len(v) - 1, int((i + 0.5) * len(v) / B))]
               for i in range(B)]
    centers = sorted(set(centers))
    if len(centers) < 2:
        raise ValueError("chronos: the samples are too concentrated "
                         "to form bins")
    edges = [0.5 * (centers[i] + centers[i + 1])
             for i in range(len(centers) - 1)]
    return {"centers": centers, "edges": edges,
            "n_bins": len(centers), "scheme": "quantile",
            "range": (centers[0], centers[-1]),
            "caveat": "fitted to the TRAINING distribution; an unseen "
                      "dataset may fall where there are no bins"}


def quantize(x, bins):
    r"""Map real values to bin indices, reporting what was clipped.

    Values outside :math:`[c_1, c_B]` are assigned to the boundary
    bin. The count is returned because that clipping is the paper's
    stated limitation for trending series, not a detail.
    """
    v = [float(q) for q in k.vec(x)]
    c, e = bins["centers"], bins["edges"]
    out, clipped = [], 0
    for q in v:
        if q < c[0]:
            clipped += 1
        elif q > c[-1]:
            clipped += 1
        j = 0
        while j < len(e) and q >= e[j]:
            j += 1
        out.append(j)
    return {"tokens": out, "n_clipped": clipped,
            "clipped_fraction": clipped / float(len(v)),
            "in_range": clipped == 0,
            "note": "predictions are confined to [c_1, c_B]; a strong "
                    "trend leaves that interval and cannot be "
                    "represented"}


def dequantize(tokens, bins):
    r"""Map bin indices back to their centres."""
    c = bins["centers"]
    out = []
    for t in tokens:
        j = int(t)
        if j in (PAD, EOS):
            continue
        if not 0 <= j < len(c):
            raise ValueError("chronos: token %d is outside the "
                             "vocabulary of %d bins" % (j, len(c)))
        out.append(c[j])
    return out


def tokenize(x, bins, context=None, add_eos=True, pad_to=None):
    r"""Scale, quantise, and append the special tokens."""
    sc = mean_scale(x, context=context)
    qz = quantize(sc["scaled"], bins)
    toks = list(qz["tokens"])
    if add_eos:
        toks.append(EOS)
    if pad_to is not None and len(toks) < int(pad_to):
        toks = [PAD] * (int(pad_to) - len(toks)) + toks
    return RichResult(payload={
        "estimate": toks, "tokens": toks, "scale": sc["scale"],
        "n_clipped": qz["n_clipped"],
        "clipped_fraction": qz["clipped_fraction"],
        "vocab_size": bins["n_bins"] + 2,
        "method": "Chronos tokenisation: mean scaling then uniform "
                  "quantisation; Ansari et al. (2024) Sec. 3.1",
        "ignores": "time and frequency features, deliberately",
    })


def detokenize(tokens, bins, scale):
    r"""Undo tokenisation: centres back to the original units."""
    return [q * float(scale) for q in dequantize(tokens, bins)]


def forecast_summary(token_probs, bins, quantiles=(0.1, 0.5, 0.9)):
    r"""Summaries of the predicted distribution over bins.

    The model emits a categorical distribution, so a point forecast is
    a *choice* made afterwards. Cross-entropy training knows nothing
    about bin order, so nothing here assumes the distribution is
    unimodal.
    """
    p = [float(q) for q in k.vec(token_probs)]
    c = bins["centers"]
    if len(p) != len(c):
        raise ValueError("chronos: %d probabilities for %d bins"
                         % (len(p), len(c)))
    tot = sum(p)
    if tot <= _EPS:
        raise ValueError("chronos: the predicted distribution has no "
                         "mass")
    p = [q / tot for q in p]
    mean = sum(p[i] * c[i] for i in range(len(c)))
    out = {}
    for qq in quantiles:
        acc, pick = 0.0, c[-1]
        for i in range(len(c)):
            acc += p[i]
            if acc >= float(qq):
                pick = c[i]
                break
        out[float(qq)] = pick
    return {"mean": mean, "quantiles": out,
            "mode": c[max(range(len(p)), key=lambda i: p[i])],
            "note": "cross-entropy training does not know bins are "
                    "ordered; the model must learn that neighbouring "
                    "bins are similar"}


def cheatsheet():
    return ("chronos: time series as a LANGUAGE. Mean scaling with "
            "m = 0 and s = mean|x| over the context -- m = 0 means "
            "ZERO MAPS TO ZERO, which matters because zeros are "
            "usually real. Uniform bins, edges exactly midway; "
            "quantile bins rejected because unseen datasets differ "
            "from the training CDF. Predictions confined to "
            "[c_1, c_B], so strong TRENDS cannot be represented. "
            "Calendar and frequency features deliberately ignored. "
            "Cross-entropy loss, so bin ORDER is not given to the "
            "model.")


# compact alias per ledger/NAMING.md
chronosforecast = tokenize
