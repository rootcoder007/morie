# morie.fn -- function file (rootcoder007/morie)
r"""MSM for a continuous-dose treatment.

Three published routes to the dose-response curve, all here, because
the sources give three and they answer slightly different questions.

**Stabilized IP weighting** (default). Robins, Hernan & Brumback (2000)
Sec. 4 extends the marginal structural model to a continuous exposure by
replacing the probability ratio with a density ratio,

.. math:: SW = \frac{f(A)}{f(A \mid L)},

and fitting the MSM in the weighted pseudo-population. The stub this
replaces had the ratio upside down -- its docstring said
"weight by f(A|H)/f_marg(A)" -- and then returned ``mean(y)``.

**Subclassification on the generalized propensity score.** Imai & van
Dyk (2004) show the propensity score generalises to arbitrary treatment
regimes as the conditional density evaluated at the observed dose, and
that one may subclassify on the *parameter* of that density rather than
weight by it. Their estimator splits the sample into strata of the
fitted dose mean, fits the dose-response within each, and averages,
which avoids the density ratio entirely and so avoids its tails.

**The dose-response function.** Hirano & Imbens (2004) estimate
:math:`\mu(a) = E[Y^a]` by regressing Y on the dose and the GPS jointly
and then averaging the fitted surface over the GPS distribution at each
dose. This is the only one of the three that returns a curve rather
than a slope.

Weighting is the default because it is the one the ledger's cited
source specifies, and because it is the only one of the three that
generalises unchanged to a treatment *history*. Its weakness is real
and is reported, not hidden: for Gaussian models the stabilized weight
has finite variance only when the treatment model's residual variance
exceeds half the marginal variance, and ``finite_variance`` says which
side of that line the data fall on.

References
----------
Robins, J. M., Hernan, M. A. & Brumback, B. (2000) "Marginal structural
models and causal inference in epidemiology", *Epidemiology* 11(5),
550-560, doi:10.1097/00001648-200009000-00011.

Imai, K. & van Dyk, D. A. (2004) "Causal inference with general
treatment regimes: generalizing the propensity score", *Journal of the
American Statistical Association* 99(467), 854-866,
doi:10.1198/016214504000001187.

Hirano, K. & Imbens, G. W. (2004) "The propensity score with continuous
treatments", in Gelman, A. & Meng, X.-L. (eds.), *Applied Bayesian
Modeling and Causal Inference from Incomplete-Data Perspectives*,
Wiley, 73-84, doi:10.1002/0470090456.ch7.

Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*,
Chapman & Hall/CRC, Sec. 12.3 for the stabilized weights and their
mean-1 diagnostic.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["generalized_treatment_msm", "gps_subclassify",
           "dose_response_curve"]

_METHODS = ("weight", "subclassify", "doseresponse")


def generalized_treatment_msm(y, A, H, method="weight", degree=1,
                              n_strata=5, doses=None, trim=None):
    r"""Dose-response MSM for a continuous treatment.

    Parameters
    ----------
    y : array-like
        Outcome.
    A : array-like
        Continuous dose.
    H : array-like
        Confounders, n-by-p.
    method : {"weight", "subclassify", "doseresponse"}
        Which of the three published routes to use.
    degree : int
        Degree of the polynomial in the dose that the MSM fits. 1 is
        the linear dose-response of the Robins-Hernan-Brumback example.
    n_strata : int
        Number of generalized-propensity-score strata, Imai & van Dyk.
    doses : array-like, optional
        Where to evaluate the Hirano-Imbens dose-response curve.

    Returns
    -------
    RichResult
        ``estimate`` is the linear dose coefficient for the first two
        methods and the curve's average derivative for the third.

    Examples
    --------
    A confounded dose whose true effect is 1.5 per unit::

        r = generalized_treatment_msm(y, dose, L)
        r["estimate"]
    """
    if method not in _METHODS:
        raise ValueError("generalized_treatment_msm: method must be one of "
                         "%r, got %r" % (_METHODS, method))
    deg = int(degree)
    if deg < 1:
        raise ValueError("generalized_treatment_msm: degree must be at "
                         "least 1, got %r" % (degree,))
    yv = k.vec(y)
    av = k.vec(A)
    n = len(yv)
    if len(av) != n:
        raise ValueError("generalized_treatment_msm: %d outcomes but %d "
                         "doses" % (n, len(av)))
    if len(set(av)) < 3:
        raise ValueError(
            "generalized_treatment_msm: the dose takes %d distinct values; "
            "this is the continuous-treatment estimator and a binary or "
            "near-binary exposure belongs in a binary MSM"
            % len(set(av)))

    if method == "weight":
        w, info = k.ip_weights(av, H, kind="normal", stabilize=True,
                               trim=trim)
        X = [[av[i] ** d for d in range(1, deg + 1)] for i in range(n)]
        fit = k.wls(X, yv, w)
        crude = k.wls(X, yv, [1.0] * n)
        return RichResult(payload={
            "estimate": fit["coef"][1],
            "se": fit["se"][1],
            "coef": fit["coef"], "vcov": fit["vcov"],
            "crude": crude["coef"][1],
            "weights": w,
            "mean_weight": info["mean_weight"],
            "max_weight": info["max_weight"],
            "effective_sample_size": info["effective_sample_size"],
            "finite_variance": info["finite_variance"],
            "variance_ratio": info["variance_ratio"],
            "gps": info["denominator"],
            "degree": deg, "n": n,
            "method": "stabilized IP weighting for a continuous dose, "
                      "Robins, Hernan & Brumback (2000)",
        })

    if method == "subclassify":
        out = gps_subclassify(yv, av, H, n_strata=n_strata, degree=deg)
        out["method"] = ("generalized propensity score subclassification, "
                         "Imai & van Dyk (2004)")
        return RichResult(payload=out)

    out = dose_response_curve(yv, av, H, doses=doses, degree=deg)
    out["method"] = ("dose-response function via the GPS, "
                     "Hirano & Imbens (2004)")
    return RichResult(payload=out)


def gps_subclassify(y, A, H, n_strata=5, degree=1):
    """Imai & van Dyk (2004): stratify on the fitted dose mean.

    Their point is that the generalized propensity score is a function
    of the treatment model's *parameter*, so subclassifying on that
    parameter balances the confounders without ever forming a density
    ratio. Within-stratum slopes are pooled by stratum size, which is
    the estimator's standardisation step.
    """
    yv = k.vec(y)
    av = k.vec(A)
    n = len(yv)
    J = int(n_strata)
    if J < 2:
        raise ValueError("gps_subclassify: need at least 2 strata, got %r"
                         % (n_strata,))
    if n < 4 * J:
        raise ValueError(
            "gps_subclassify: %d observations cannot support %d strata; "
            "each needs enough points to fit a degree-%d dose model"
            % (n, J, degree))
    _, info = k.treatment_density(av, H, kind="normal")
    mu = list(info["mu"])
    order = sorted(range(n), key=lambda i: mu[i])
    edges = [int(round(j * n / float(J))) for j in range(J + 1)]
    slopes, sizes, ses = [], [], []
    for j in range(J):
        idx = order[edges[j]:edges[j + 1]]
        if len(idx) < degree + 2:
            continue
        Xs = [[av[i] ** d for d in range(1, int(degree) + 1)] for i in idx]
        ys = [yv[i] for i in idx]
        f = k.wls(Xs, ys, [1.0] * len(idx))
        slopes.append(f["coef"][1])
        ses.append(f["se"][1])
        sizes.append(len(idx))
    if not slopes:
        raise ValueError("gps_subclassify: every stratum was too small to "
                         "fit; reduce n_strata")
    tot = float(sum(sizes))
    est = sum(slopes[j] * sizes[j] for j in range(len(slopes))) / tot
    var = sum((sizes[j] / tot) ** 2 * ses[j] ** 2
              for j in range(len(slopes)))
    return {"estimate": est, "se": math.sqrt(var) if var > 0 else
            float("nan"),
            "stratum_slopes": slopes, "stratum_sizes": sizes,
            "stratum_se": ses, "gps_mean": mu, "n_strata": len(slopes),
            "n": n, "degree": int(degree)}


def dose_response_curve(y, A, H, doses=None, degree=1):
    r"""Hirano & Imbens (2004): E[Y^a] by averaging over the GPS.

    Regress Y on the dose and the GPS jointly, then for each dose a
    average the fitted surface over the distribution of the GPS
    *evaluated at that dose* -- not over the observed GPS, which is the
    step people get wrong. Returns the curve and its average derivative.
    """
    yv = k.vec(y)
    av = k.vec(A)
    n = len(yv)
    dens, info = k.treatment_density(av, H, kind="normal")
    mu = list(info["mu"])
    s2 = float(info["sigma2"])
    deg = int(degree)

    def gps_at(a, i):
        r = a - mu[i]
        return math.exp(-0.5 * r * r / s2) / math.sqrt(2.0 * math.pi * s2)

    X = [[av[i] ** d for d in range(1, deg + 1)]
         + [dens[i], dens[i] * dens[i], av[i] * dens[i]]
         for i in range(n)]
    fit = k.wls(X, yv, [1.0] * n)
    b = fit["coef"]

    if doses is None:
        lo, hi = min(av), max(av)
        doses = [lo + (hi - lo) * t / 20.0 for t in range(21)]
    doses = [float(v) for v in k.vec(doses)]
    curve = []
    for a in doses:
        tot = 0.0
        for i in range(n):
            r = gps_at(a, i)
            row = [1.0] + [a ** d for d in range(1, deg + 1)] \
                + [r, r * r, a * r]
            tot += sum(b[j] * row[j] for j in range(len(b)))
        curve.append(tot / n)
    slopes = [(curve[t + 1] - curve[t]) / (doses[t + 1] - doses[t])
              for t in range(len(doses) - 1)
              if doses[t + 1] != doses[t]]
    est = sum(slopes) / len(slopes) if slopes else float("nan")
    return {"estimate": est, "se": float("nan"), "doses": doses,
            "curve": curve, "slopes": slopes, "coef": b,
            "gps": dens, "n": n, "degree": deg}


def cheatsheet():
    return ("gentmt: continuous-dose MSM. weight = SW = f(A)/f(A|L) "
            "(Robins-Hernan-Brumback 2000, default); subclassify = GPS "
            "strata (Imai-van Dyk 2004); doseresponse = E[Y^a] curve "
            "(Hirano-Imbens 2004). Reports finite_variance.")


# compact alias per ledger/NAMING.md
generalizedtreatmentmsm = generalized_treatment_msm
