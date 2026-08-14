# morie.fn -- function file (rootcoder007/morie)
r"""OpenCLIP: scaling laws you can actually reproduce.

Scaling laws let you predict what a model will do before paying for
it, but the CLIP-scale ones had been measured on **private** data and
models. OpenCLIP re-runs the measurement on the public LAION dataset
with an open implementation, up to two billion image-text pairs, and
finds **power law** scaling for zero-shot classification, retrieval,
linear probing and fine-tuning alike.

**The finding that matters is not the exponent but that it moves.**
OpenAI's and OpenCLIP's models exhibit **different scaling behaviour
despite identical architectures and similar recipes** -- so the
training *distribution* is part of the law, and an exponent measured
on one corpus does not transfer to another. ``compare_scaling`` exists
to make that comparison, since a single fitted exponent quietly
implies a universality the paper explicitly denies.

**Fitting is a straight line in log-log space.** With
:math:`E = \beta C^{-\alpha}`, :math:`\log E = \log\beta - \alpha\log
C`, so ordinary least squares on the logs recovers
:math:`(\alpha, \beta)` exactly for noiseless data -- which is what the
anchor checks, rather than checking that a curve looks like a curve.

**Extrapolation is where a scaling law earns its keep and where it
lies.** ``predict`` therefore reports how far beyond the fitted range
the query sits: a prediction one decade out is a different object from
an interpolation.

References
----------
Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G.,
Gordon, C., Schuhmann, C., Schmidt, L. & Jitsev, J. (2023)
"Reproducible scaling laws for contrastive language-image learning",
*Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR 2023)*, 2818-2829, arXiv:2212.07143. The
abstract and Sec. 1: that previous work on scaling laws primarily used
private data and models or focused on uni-modal learning; the
investigation of scaling laws for contrastive language-image
pre-training with the public LAION dataset and the open-source
OpenCLIP repository; experiments on models trained on up to two
billion image-text pairs identifying POWER LAW scaling for zero-shot
classification, retrieval, linear probing and end-to-end fine-tuning;
and the finding that the training distribution plays a key role, as
the OpenAI and OpenCLIP models exhibit different scaling behaviour
despite identical model architectures and similar training recipes.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal,
S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G. &
Sutskever, I. (2021) "Learning Transferable Visual Models From Natural
Language Supervision", *ICML 2021*, PMLR 139, 8748-8763,
arXiv:2103.00020. CLIP itself.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B.,
Child, R., Gray, S., Radford, A., Wu, J. & Amodei, D. (2020)
"Scaling Laws for Neural Language Models", arXiv:2001.08361. The
power-law form.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["total_compute", "fit_power_law", "predict",
           "compare_scaling", "infonce"]

_EPS = 1e-12


def total_compute(samples_seen, model_params):
    r"""Samples seen times parameters -- the axis the law is in.

    Epochs are the wrong unit when the dataset size is itself being
    varied; samples SEEN is what the models actually consumed.
    """
    s, p = float(samples_seen), float(model_params)
    if s <= 0.0 or p <= 0.0:
        raise ValueError("opnclp: both quantities must be positive")
    return {"compute": s * p, "samples_seen": s, "params": p,
            "gmac_scale": s * p / 1e9}


def fit_power_law(x, y):
    r"""Least squares on the logs of :math:`E = \beta C^{-\alpha}`.

    Exact for noiseless data, which is what makes it checkable.
    """
    X = [float(v) for v in k.vec(x)]
    Y = [float(v) for v in k.vec(y)]
    if len(X) != len(Y):
        raise ValueError("opnclp: %d x values but %d y values"
                         % (len(X), len(Y)))
    if len(X) < 2:
        raise ValueError("opnclp: at least 2 points are needed")
    if any(v <= 0.0 for v in X) or any(v <= 0.0 for v in Y):
        raise ValueError("opnclp: a power law is fitted on the logs, "
                         "so both axes must be strictly positive")
    lx = [math.log(v) for v in X]
    ly = [math.log(v) for v in Y]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    if sxx <= _EPS:
        raise ValueError("opnclp: every x is the same, so no slope "
                         "is identified")
    sxy = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    slope = sxy / sxx
    inter = my - slope * mx
    pred = [inter + slope * v for v in lx]
    ss_res = sum((ly[i] - pred[i]) ** 2 for i in range(n))
    ss_tot = sum((v - my) ** 2 for v in ly)
    return {"alpha": -slope, "beta": math.exp(inter),
            "slope": slope, "r_squared": 1.0 - ss_res / ss_tot
            if ss_tot > _EPS else 1.0,
            "range": (min(X), max(X)), "n": n}


def predict(fit, compute):
    r"""Apply the law -- and say how far outside the fit it reaches."""
    c = float(compute)
    if c <= 0.0:
        raise ValueError("opnclp: compute must be positive")
    lo, hi = fit["range"]
    if c < lo:
        decades = math.log10(lo / c)
    elif c > hi:
        decades = math.log10(c / hi)
    else:
        decades = 0.0
    return {"value": fit["beta"] * c ** (-fit["alpha"]),
            "extrapolation_decades": decades,
            "interpolated": decades == 0.0,
            "note": "an extrapolation is a different claim from an "
                    "interpolation, so the distance is reported"}


def compare_scaling(x_a, y_a, x_b, y_b, label_a="A", label_b="B"):
    r"""Two corpora, same architecture: do the exponents agree?

    The paper's central caution -- identical architectures and similar
    recipes gave DIFFERENT scaling behaviour, so the exponent belongs
    to the data as much as to the model.
    """
    fa = fit_power_law(x_a, y_a)
    fb = fit_power_law(x_b, y_b)
    d = abs(fa["alpha"] - fb["alpha"])
    return RichResult(payload={
        "estimate": d, "alpha_gap": d,
        label_a: fa, label_b: fb,
        "same_law": d < 0.01,
        "method": "power-law comparison across training "
                  "distributions; Cherti et al. (2023)",
        "note": "a single exponent implies a universality the paper "
                "denies; this is where the distribution shows up",
    })


def infonce(image_embeddings, text_embeddings, temperature=0.07):
    r"""The contrastive objective, symmetric in the two directions."""
    I = [[float(v) for v in r] for r in k.mat(image_embeddings)]
    T = [[float(v) for v in r] for r in k.mat(text_embeddings)]
    n = len(I)
    if len(T) != n:
        raise ValueError("opnclp: %d images but %d texts"
                         % (n, len(T)))
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("opnclp: the temperature must be positive")

    def nrm(v):
        m = math.sqrt(sum(x * x for x in v))
        if m <= _EPS:
            raise ValueError("opnclp: a zero embedding has no "
                             "direction")
        return [x / m for x in v]

    Iu = [nrm(v) for v in I]
    Tu = [nrm(v) for v in T]
    S = [[sum(Iu[i][a] * Tu[j][a] for a in range(len(Iu[0]))) / t
          for j in range(n)] for i in range(n)]

    def ce(rows):
        tot = 0.0
        for i in range(n):
            m = max(rows[i])
            z = sum(math.exp(v - m) for v in rows[i])
            tot += -(rows[i][i] - m - math.log(z))
        return tot / n

    li = ce(S)
    lt = ce([[S[j][i] for j in range(n)] for i in range(n)])
    return {"loss": 0.5 * (li + lt), "image_to_text": li,
            "text_to_image": lt, "logits": S,
            "note": "symmetric, so neither modality is the anchor"}


def cheatsheet():
    return ("opnclp: CLIP-scale laws had been measured on PRIVATE data "
            "and models; re-run on public LAION with an open "
            "implementation, up to 2B pairs, and the scaling is a "
            "POWER LAW across zero-shot classification, retrieval, "
            "linear probing and fine-tuning. The key finding is that "
            "the exponent MOVES: OpenAI and OpenCLIP models scale "
            "differently despite identical architectures and similar "
            "recipes, so the training DISTRIBUTION is part of the law. "
            "Fit by least squares on the logs; report how many decades "
            "beyond the fitted range a prediction reaches.")


# compact alias per ledger/NAMING.md
openclipscaling = fit_power_law
