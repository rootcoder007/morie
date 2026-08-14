# morie.fn -- function file (rootcoder007/morie)
r"""CV-TMLE for nonpathwise differentiable target parameters.

TMLE builds efficient substitution estimators for **pathwise
differentiable** parameters. Many parameters are not: a density at a
point, or a regression curve evaluated at a single :math:`w`, in a
nonparametric model. There is no efficient influence curve to solve
against, and no root-:math:`n` estimator.

**What is normally done, and why it disappoints.** One picks a
specific estimator -- a kernel with a chosen bandwidth, say -- under a
specific smoothness assumption, and derives a limit distribution from
it. The estimator is then tied to an assumed smoothness it cannot
verify, and if the truth is smoother it is beaten by an estimator that
adapts.

**The construction.** Approximate the nonpathwise parameter by a
**smoothed** one that *is* pathwise differentiable -- a kernel average
of the density over a window :math:`h`, whose efficient influence
curve exists -- and estimate that with CV-TMLE. Then select :math:`h`
data-adaptively. The result converges at the adaptive optimal rate
implied by the true unknown smoothness while still supplying formal
inference, which is what neither the fixed-bandwidth estimator nor an
unsmoothed plug-in can do.

**The bias-variance trade is explicit, not hidden.** The smoothed
parameter differs from the target by an approximation bias of order
:math:`h^s` for smoothness :math:`s`, while its variance grows like
:math:`1/(nh)`. Inference is for the *smoothed* parameter, and the
interval covers the true one only when the bias is dominated --
``smoothing_bias`` reports the ratio so the gap is visible rather than
assumed away.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 25 (van der
Laan, Bibaut & Luedtke): TMLE was developed for efficient substitution
estimators of pathwise differentiable target parameters, while many
parameters -- a density or regression curve at a single point in a
nonparametric model -- are nonpathwise differentiable; the usual
recourse to a specific estimator under specific smoothness
assumptions, which does not adapt to the true unknown smoothness and
can be outperformed by an adaptive estimator; and the fully adaptive
estimator converging at the adaptive optimal rate implied by the
unknown smoothness while still providing formal inference, using
CV-TMLE for a data-adaptively selected smooth approximation of the
nonpathwise differentiable parameter, integrating efficiency theory
with super learning.

Bibaut, A. F. & van der Laan, M. J. (2019) "Fast rates for empirical
risk minimization over cadlag functions with bounded sectional
variation norm", arXiv:1907.09244.

Lepski, O. V. & Spokoiny, V. G. (1997) "Optimal pointwise adaptive
methods in nonparametric estimation", *Annals of Statistics* 25(6),
2512-2546, doi:10.1214/aos/1030741083. Adaptive bandwidth selection.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["kernel_smooth", "smoothed_parameter", "smoothing_bias",
           "select_bandwidth", "cv_tmle_smoothed"]

_EPS = 1e-12
_KERNELS = ("epanechnikov", "gaussian", "uniform")


def kernel_smooth(u, kernel="epanechnikov"):
    r"""Kernel weight at scaled distance :math:`u`."""
    if kernel not in _KERNELS:
        raise ValueError("tlcvnp: kernel must be one of %s, got %r"
                         % (", ".join(_KERNELS), kernel))
    v = float(u)
    if kernel == "epanechnikov":
        return 0.75 * (1.0 - v * v) if abs(v) <= 1.0 else 0.0
    if kernel == "uniform":
        return 0.5 if abs(v) <= 1.0 else 0.0
    return math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)


def smoothed_parameter(X, x0, h, kernel="epanechnikov"):
    r"""The pathwise differentiable stand-in for the density at
    :math:`x_0`.

    :math:`\Psi_h(P) = \int \frac{1}{h}K\!\big(\frac{x-x_0}{h}\big)
    dP(x)` -- an average, hence pathwise differentiable, unlike the
    density itself.
    """
    v = [float(q) for q in k.vec(X)]
    hh = float(h)
    if hh <= 0.0:
        raise ValueError("tlcvnp: the bandwidth must be positive")
    n = len(v)
    val = sum(kernel_smooth((v[i] - float(x0)) / hh, kernel)
              for i in range(n)) / (n * hh)
    ic = [kernel_smooth((v[i] - float(x0)) / hh, kernel) / hh - val
          for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((q - m) ** 2 for q in ic) / (n - 1) / n)
    return {"psi_h": val, "se": se, "h": hh, "n": n,
            "influence_curve": ic,
            "note": "the SMOOTHED parameter is pathwise "
                    "differentiable; the density at a point is not"}


def smoothing_bias(true_density, x0, h, smoothness=2.0):
    r"""Approximation bias against sampling error.

    Bias is :math:`O(h^s)` and the standard error :math:`O(1/\sqrt{nh})`
    -- the interval covers the TRUE parameter only where the first is
    dominated by the second.
    """
    hh, s = float(h), float(smoothness)
    if hh <= 0.0 or s <= 0.0:
        raise ValueError("tlcvnp: bandwidth and smoothness must be "
                         "positive")
    return {"bias_order": hh ** s, "h": hh, "smoothness": s,
            "note": "inference is for the SMOOTHED parameter; it "
                    "transfers to the target only when the "
                    "approximation bias is dominated"}


def select_bandwidth(X, x0, bandwidths, kernel="epanechnikov",
                     criterion="lepski", C=1.0):
    r"""Data-adaptive bandwidth.

    The Lepski rule takes the smallest :math:`h` whose estimate stays
    within the confidence band of every larger one -- selecting on the
    data rather than on an assumed smoothness, which is the entire
    point.
    """
    hs = sorted(float(v) for v in bandwidths)
    if not hs:
        raise ValueError("tlcvnp: no bandwidths given")
    if criterion not in ("lepski", "smallest_se"):
        raise ValueError("tlcvnp: criterion must be lepski or "
                         "smallest_se, got %r" % (criterion,))
    fits = [smoothed_parameter(X, x0, h, kernel) for h in hs]
    if criterion == "smallest_se":
        j = min(range(len(hs)), key=lambda i: fits[i]["se"])
        return {"h": hs[j], "fit": fits[j], "criterion": criterion}
    chosen = len(hs) - 1
    for i in range(len(hs)):
        ok = True
        for j in range(i + 1, len(hs)):
            if abs(fits[i]["psi_h"] - fits[j]["psi_h"]) > \
                    float(C) * (fits[i]["se"] + fits[j]["se"]):
                ok = False
                break
        if ok:
            chosen = i
            break
    return {"h": hs[chosen], "fit": fits[chosen],
            "criterion": criterion,
            "all": [(hs[i], fits[i]["psi_h"], fits[i]["se"])
                    for i in range(len(hs))],
            "note": "the smallest bandwidth consistent with every "
                    "larger one"}


def cv_tmle_smoothed(X, x0, bandwidths, kernel="epanechnikov",
                     V=5, seed=0):
    r"""CV-TMLE of the smoothed parameter at the selected bandwidth.

    Selection happens on the training split and estimation on the
    held-out one, so the bandwidth is fixed conditional on the
    training data.
    """
    v = [float(q) for q in k.vec(X)]
    n = len(v)
    rng = np.random.default_rng(seed)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(float(rng.uniform()) * (i + 1)) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    folds = [sorted(idx[f::int(V)]) for f in range(int(V))]
    ests, ics, hs = [], [0.0] * n, []
    for f in folds:
        tr = [v[i] for i in range(n) if i not in set(f)]
        sel = select_bandwidth(tr, x0, bandwidths, kernel)
        hs.append(sel["h"])
        est = smoothed_parameter([v[i] for i in f], x0, sel["h"],
                                 kernel)
        ests.append(est["psi_h"])
        for a, i in enumerate(f):
            ics[i] = est["influence_curve"][a]
    psi = sum(ests) / len(ests)
    m = sum(ics) / n
    se = math.sqrt(sum((q - m) ** 2 for q in ics) / (n - 1) / n)
    return RichResult(payload={
        "estimate": psi, "psi": psi, "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "bandwidths": hs, "fold_estimates": ests, "V": int(V),
        "method": "CV-TMLE for a data-adaptively smoothed nonpathwise "
                  "parameter; van der Laan & Rose (2018) Chap. 25",
        "note": "adapts to the unknown smoothness instead of assuming "
                "it, and still supplies formal inference",
    })


def cheatsheet():
    return ("tlcvnp: a density or regression curve AT A POINT is "
            "NONpathwise differentiable -- no efficient influence "
            "curve, no root-n estimator. The usual fix picks a "
            "bandwidth under an assumed smoothness and is beaten by "
            "anything adaptive. Instead approximate the target by a "
            "SMOOTHED parameter that IS pathwise differentiable, "
            "estimate it by CV-TMLE, and choose the bandwidth from the "
            "data (Lepski). The bias is O(h^s) and the standard error "
            "O(1/sqrt(nh)): inference is for the smoothed parameter "
            "and transfers only when the bias is dominated.")


# compact alias per ledger/NAMING.md
cvtmlenonpathwise = cv_tmle_smoothed
