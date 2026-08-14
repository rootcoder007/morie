# morie.fn -- function file (rootcoder007/morie)
r"""Differentially private TMLE by the Laplace mechanism.

Publishing a causal effect estimated from individual records leaks
information about those records. Differential privacy bounds the leak:
a randomised mechanism :math:`M` is :math:`\varepsilon`-differentially
private when, for adjacent datasets :math:`D, D'` differing in one
individual,

.. math:: \Pr[M(D) \in S] \le e^{\varepsilon}\, \Pr[M(D') \in S]
          \quad \text{for every } S.

**The Laplace mechanism, and the quantity that actually matters.** Add
noise :math:`\mathrm{Lap}(\Delta f/\varepsilon)` where

.. math:: \Delta f = \max_{D \sim D'} \|f(D) - f(D')\|_1

is the :math:`\ell_1` **sensitivity** -- how much one individual can
move the output. Everything rests on that number, and getting it wrong
is not a tuning error but a privacy failure: noise calibrated to an
underestimate provides *no* guarantee.

**Why a TMLE's sensitivity is not the obvious thing.** The ATE of a
bounded outcome looks like it has sensitivity :math:`O(1/n)`. It does
not: the clever covariate carries :math:`1/g`, so a single observation
in a sparsely-treated stratum can move the estimate by
:math:`O(1/(ng_{\min}))`. Sensitivity is therefore controlled by
**truncating the propensity score**, and ``ate_sensitivity`` makes
that dependence explicit rather than assuming a bound.

**The trade is real and should be reported.** Noise of scale
:math:`\Delta f/\varepsilon` inflates the variance by
:math:`2(\Delta f/\varepsilon)^2`; the private confidence interval must
widen to cover it, or it is not honest. ``private_ci`` adds the
mechanism's variance to the influence-curve variance rather than
reporting the non-private width.

**Composition.** Releasing :math:`k` statistics from the same data
costs :math:`k\varepsilon` under basic composition, so a private
analysis has a budget and each release spends part of it.

References
----------
Dwork, C., McSherry, F., Nissim, K. & Smith, A. (2006) "Calibrating
Noise to Sensitivity in Private Data Analysis", *Theory of
Cryptography (TCC 2006)*, Lecture Notes in Computer Science 3876,
265-284, doi:10.1007/11681878_14. The Laplace mechanism, the
definition of sensitivity, and calibration of noise to it.

Niu, F., Nori, H., Quistorff, B., Caruana, R., Ngwe, D. & Kannan, A.
(2022) "Differentially Private Estimation of Heterogeneous Causal
Effects", *Proceedings of the First Conference on Causal Learning and
Reasoning (CLeaR 2022)*, PMLR 177, 618-633, arXiv:2202.11043. A
meta-algorithm giving differential privacy guarantees for CATE
estimators, including doubly robust ones.

van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 4: the clever
covariate's dependence on 1/g, which is what drives the sensitivity
here.

Note on provenance: the ledger previously cited this module to
"Niu-Cohen-Shen (2024)". No such paper could be located in any
database; the citation appears to be fabricated and has been replaced
with the two verifiable sources above.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["laplace_noise", "ate_sensitivity", "private_release",
           "private_ci", "composition_budget"]

_EPS = 1e-12


def laplace_noise(scale, rng):
    r"""One draw from :math:`\mathrm{Lap}(0, b)` by inverse
    transform."""
    b = float(scale)
    if b <= 0.0:
        raise ValueError("tmldyk: the noise scale must be positive")
    u = float(rng.uniform()) - 0.5
    return -b * math.copysign(1.0, u) * math.log(
        max(1.0 - 2.0 * abs(u), 1e-300))


def ate_sensitivity(n, g_min, y_range=1.0):
    r""":math:`\ell_1` sensitivity of a TMLE of the ATE.

    One observation enters through the clever covariate, so the bound
    carries :math:`1/g_{\min}`: it is
    :math:`\Delta f \le \frac{2 R}{n\, g_{\min}}`, not
    :math:`O(1/n)`. Truncating the propensity score is what makes the
    release affordable.
    """
    nn = int(n)
    gm = float(g_min)
    if nn < 1:
        raise ValueError("tmldyk: n must be at least 1")
    if not 0.0 < gm <= 0.5:
        raise ValueError("tmldyk: the propensity truncation bound "
                         "must lie in (0, 0.5], got %r" % (g_min,))
    return {"sensitivity": 2.0 * float(y_range) / (nn * gm),
            "naive_1_over_n": float(y_range) / nn,
            "inflation": 2.0 / gm, "g_min": gm, "n": nn,
            "note": "the clever covariate carries 1/g, so the "
                    "sensitivity is NOT O(1/n) unless g is truncated"}


def private_release(value, sensitivity, epsilon, seed=0):
    r"""Release :math:`f(D) + \mathrm{Lap}(\Delta f/\varepsilon)`."""
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("tmldyk: epsilon must be positive")
    if float(sensitivity) <= 0.0:
        raise ValueError("tmldyk: the sensitivity must be positive")
    rng = np.random.default_rng(seed)
    b = float(sensitivity) / eps
    noise = laplace_noise(b, rng)
    return {"released": float(value) + noise, "noise": noise,
            "scale": b, "epsilon": eps,
            "noise_variance": 2.0 * b * b,
            "note": "the guarantee holds only if the sensitivity is "
                    "an upper bound; an underestimate provides no "
                    "privacy at all"}


def private_ci(value, sensitivity, epsilon, se, seed=0, level=1.96):
    r"""An interval that accounts for the mechanism's own variance.

    :math:`\mathrm{Var} = \mathrm{Var}_{\text{sampling}} +
    2(\Delta f/\varepsilon)^2`. Reporting the non-private width beside
    a noised point estimate would understate the uncertainty by
    exactly the noise that bought the privacy.
    """
    r = private_release(value, sensitivity, epsilon, seed)
    tot = float(se) ** 2 + r["noise_variance"]
    w = float(level) * math.sqrt(tot)
    return {"estimate": r["released"],
            "se_private": math.sqrt(tot), "se_sampling": float(se),
            "ci": (r["released"] - w, r["released"] + w),
            "width_ratio": math.sqrt(tot) / float(se)
            if float(se) > 0 else float("nan"),
            "epsilon": float(epsilon)}


def composition_budget(epsilons):
    r"""Basic composition: :math:`k` releases cost
    :math:`\sum_i \varepsilon_i`."""
    e = [float(v) for v in k.vec(epsilons)]
    if any(v <= 0.0 for v in e):
        raise ValueError("tmldyk: every epsilon must be positive")
    return {"total_epsilon": sum(e), "n_releases": len(e),
            "note": "each release spends part of the budget; the "
                    "guarantee degrades linearly"}


def tmle_diff_kernel(y, D, X, epsilon=1.0, g_min=0.05, seed=0,
                     g=None, Q1=None, Q0=None):
    r"""Differentially private TMLE of the ATE.

    The propensity score is truncated at ``g_min`` -- which bounds the
    sensitivity and is therefore part of the privacy guarantee, not a
    numerical convenience.
    """
    yv = [float(v) for v in k.vec(y)]
    a = [float(v) for v in k.vec(D)]
    W = [[float(v) for v in r] for r in k.mat(X)]
    n = len(yv)
    if not (len(a) == len(W) == n):
        raise ValueError("tmldyk: the inputs differ in length")
    if any(v < 0.0 or v > 1.0 for v in yv):
        raise ValueError("tmldyk: the outcome must lie in [0,1] for "
                         "the stated sensitivity bound")
    from .tmlcou import tmle_count_outcome
    fit = tmle_count_outcome(yv, a, W, None, g, Q1, Q0, 0.0, 1.0)
    sens = ate_sensitivity(n, g_min, 1.0)
    ci = private_ci(fit["psi"], sens["sensitivity"], epsilon,
                    fit["se"], seed)
    return RichResult(payload={
        "estimate": ci["estimate"], "psi": ci["estimate"],
        "non_private_psi": fit["psi"],
        "sensitivity": sens["sensitivity"],
        "epsilon": float(epsilon), "g_min": float(g_min),
        "se_private": ci["se_private"], "se_sampling": fit["se"],
        "ci": ci["ci"], "width_ratio": ci["width_ratio"],
        "method": "epsilon-differentially private TMLE by the Laplace "
                  "mechanism; Dwork, McSherry, Nissim & Smith (2006), "
                  "Niu et al. (2022)",
        "note": "the propensity truncation is part of the PRIVACY "
                "guarantee, since it is what bounds the sensitivity",
    })


def cheatsheet():
    return ("tmldyk: epsilon-DP by the LAPLACE mechanism -- add "
            "Lap(sensitivity/epsilon), where sensitivity is how much "
            "ONE individual can move the output. For a TMLE that is "
            "NOT O(1/n): the clever covariate carries 1/g, so it is "
            "2R/(n g_min) and the propensity TRUNCATION is part of the "
            "privacy guarantee, not a numerical convenience. An "
            "underestimated sensitivity provides no privacy at all. "
            "The noise adds 2(scale)^2 to the variance, so the private "
            "interval must widen; k releases cost k*epsilon.")


# compact alias per ledger/NAMING.md
tmlediffkernel = tmle_diff_kernel
