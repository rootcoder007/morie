# morie.fn -- function file (rootcoder007/morie)
r"""Inverse-variance weighted Mendelian randomization from summary data.

**The estimator.** Each genetic variant :math:`j` gives a ratio
estimate of the causal effect,
:math:`\hat\beta_{IVj} = \hat\beta_{Yj}/\hat\beta_{Xj}`, and the IVW
estimate is their inverse-variance weighted mean,

.. math:: \hat\beta_{IVW}
          = \frac{\sum_j \hat\beta_{IVj}\,\mathrm{var}
            (\hat\beta_{IVj})^{-1}}
                 {\sum_j \mathrm{var}(\hat\beta_{IVj})^{-1}}
          = \frac{\sum_j \hat\beta_{Yj}\hat\beta_{Xj}
            \sigma_{Yj}^{-2}}
                 {\sum_j \hat\beta_{Xj}^2 \sigma_{Yj}^{-2}} ,

the second form holding under first-order weights. That is *identically*
a weighted regression of :math:`\hat\beta_{Yj}` on
:math:`\hat\beta_{Xj}` through the origin with weights
:math:`\sigma_{Yj}^{-2}`, and the paper shows it is also the two-stage
least squares estimate when the variants are uncorrelated. Both routes
are implemented and the anchor holds them against each other to
machine precision -- an algebraic identity, so any discrepancy is a
coding error rather than a modelling choice.

**Two choices of weight.** The delta-method variance of a ratio,

.. math:: \mathrm{var}(\hat\beta_{IVj})
          = \frac{\sigma_{Yj}^2}{\hat\beta_{Xj}^2}
          + \frac{\hat\beta_{Yj}^2\sigma_{Xj}^2}{\hat\beta_{Xj}^4}
          - \frac{2\theta\hat\beta_{Yj}\sigma_{Yj}\sigma_{Xj}}
                 {\hat\beta_{Xj}^3},

keeps first *and* second order terms and a correlation
:math:`\theta` between the two association estimates. Dropping
everything but the first term gives the familiar
:math:`\sigma_{Yj}^2/\hat\beta_{Xj}^2`. The paper's finding is that
the first-order weights over-reject the null when the two association
sets come from *overlapping* samples, and it recommends the
second-order weights there; ``weights="second_order"`` takes
:math:`\theta`, which is zero in a genuine two-sample design.

**Three meta-analysis models, all kept.**

``fixed``
    All variants target the same effect.
``multiplicative``
    :math:`\hat\beta_{IVj} \sim N(\beta, \phi_M^2\sigma_{IVj}^2)`.
    Same point estimate as fixed, standard error multiplied by
    :math:`\hat\phi_M` -- the residual standard error of the same
    weighted regression. Under-dispersion is *not* passed through:
    :math:`\hat\phi_M` is floored at 1, as the paper requires, since
    a residual error below one is taken to be chance.
``additive``
    DerSimonian-Laird: heterogeneity :math:`\hat\phi_A^2` enters the
    weights, so the point estimate moves as well as the interval, and
    poorly weighted variants are relatively upweighted.

The paper's own conclusion is the default here: ``multiplicative``,
because a fixed-effect analysis over-rejects the null as soon as the
variant-specific estimates disagree.

References
----------
Burgess, S. & Bowden, J. (2015) "Integrating summarized data from
multiple genetic variants in Mendelian randomization: bias and
coverage properties of inverse-variance weighted methods",
arXiv:1512.04486 [stat.AP]. Sec. 2.1 for the ratio estimate, the
delta-method variance with first- and second-order terms (2)-(5) and
the IVW estimate (6)-(8); Sec. 2.2 for the equivalence to two-stage
least squares and to weighted regression through the origin; Sec. 2.3
and 2.4 for the fixed-effect, additive and multiplicative
random-effects models, the identity of the fixed and multiplicative
point estimates, and the instruction to floor
:math:`\hat\phi_M` at one under under-dispersion; and the abstract for
the recommendation of random-effects models and of second-order
weights under sample overlap.

DerSimonian, R. & Laird, N. (1986) "Meta-analysis in clinical trials",
*Controlled Clinical Trials* 7(3), 177-188,
doi:10.1016/0197-2456(86)90046-2, for the method-of-moments
heterogeneity estimator used by the additive model.

Lawlor, D. A., Harbord, R. M., Sterne, J. A. C., Timpson, N. & Davey
Smith, G. (2008) "Mendelian randomization: using genes as instruments
for making causal inferences in epidemiology", *Statistics in
Medicine* 27(8), 1133-1163, doi:10.1002/sim.3034, for the ratio
estimate itself.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["ratio_estimates", "ivw_variance", "ivw", "weighted_regression",
           "heterogeneity", "MODELS", "WEIGHTS"]

MODELS = ("multiplicative", "fixed", "additive")
WEIGHTS = ("first_order", "second_order")


def _check(beta_x, se_x, beta_y, se_y):
    L = len(beta_x)
    if not (L == len(se_x) == len(beta_y) == len(se_y)):
        raise ValueError("mtr2sx: the four summary vectors must have "
                         "the same length")
    if L == 0:
        raise ValueError("mtr2sx: no genetic variants supplied")
    if any(v == 0.0 for v in beta_x):
        raise ValueError("mtr2sx: a variant with zero association to "
                         "the risk factor has an undefined ratio "
                         "estimate")
    if any(v <= 0.0 for v in se_y) or any(v < 0.0 for v in se_x):
        raise ValueError("mtr2sx: standard errors must be positive")
    return L


def ratio_estimates(beta_x, beta_y):
    r"""The per-variant ratio estimate
    :math:`\hat\beta_{Yj}/\hat\beta_{Xj}`."""
    if len(beta_x) != len(beta_y):
        raise ValueError("mtr2sx: mismatched lengths")
    if any(v == 0.0 for v in beta_x):
        raise ValueError("mtr2sx: zero association to the risk factor")
    return [float(beta_y[j]) / float(beta_x[j])
            for j in range(len(beta_x))]


def ivw_variance(beta_x, se_x, beta_y, se_y, weights="first_order",
                 theta=0.0):
    r"""Delta-method variance of each ratio estimate, eqs (2)-(5)."""
    if weights not in WEIGHTS:
        raise ValueError("mtr2sx: weights must be one of %s, got %r"
                         % (", ".join(WEIGHTS), weights))
    L = _check(beta_x, se_x, beta_y, se_y)
    out = []
    for j in range(L):
        bx, by = float(beta_x[j]), float(beta_y[j])
        sy, sx = float(se_y[j]), float(se_x[j])
        v = sy * sy / (bx * bx)
        if weights == "second_order":
            v += by * by * sx * sx / (bx ** 4)
            v -= 2.0 * float(theta) * by * sy * sx / (bx ** 3)
            if v <= 0.0:
                raise ValueError("mtr2sx: the second-order variance "
                                 "for variant %d is non-positive; "
                                 "check theta" % j)
        out.append(v)
    return out


def weighted_regression(beta_x, beta_y, w):
    r"""Weighted regression through the origin -- the same estimate."""
    num = sum(w[j] * beta_x[j] * beta_y[j] for j in range(len(w)))
    den = sum(w[j] * beta_x[j] * beta_x[j] for j in range(len(w)))
    if den <= 0.0:
        raise ValueError("mtr2sx: the weighted design is degenerate")
    est = num / den
    resid = [beta_y[j] - est * beta_x[j] for j in range(len(w))]
    rss = sum(w[j] * resid[j] ** 2 for j in range(len(w)))
    dof = len(w) - 1
    return {"estimate": est, "se_fixed": math.sqrt(1.0 / den),
            "residuals": resid, "rss": rss,
            "residual_se": math.sqrt(rss / dof) if dof > 0 else 1.0}


def heterogeneity(estimates, variances, pooled):
    r"""Cochran's :math:`Q` and the DerSimonian-Laird
    :math:`\hat\phi_A^2`."""
    L = len(estimates)
    w = [1.0 / v for v in variances]
    Q = sum(w[j] * (estimates[j] - pooled) ** 2 for j in range(L))
    dof = L - 1
    if dof <= 0:
        return {"Q": Q, "df": 0, "tau2": 0.0, "I2": 0.0}
    sw = sum(w)
    sw2 = sum(v * v for v in w)
    denom = sw - sw2 / sw
    tau2 = max((Q - dof) / denom, 0.0) if denom > 0.0 else 0.0
    return {"Q": Q, "df": dof, "tau2": tau2,
            "I2": max(0.0, (Q - dof) / Q) if Q > 0.0 else 0.0}


def ivw(beta_x, se_x, beta_y, se_y, model="multiplicative",
        weights="first_order", theta=0.0):
    r"""The inverse-variance weighted causal estimate."""
    if model not in MODELS:
        raise ValueError("mtr2sx: model must be one of %s, got %r"
                         % (", ".join(MODELS), model))
    L = _check(beta_x, se_x, beta_y, se_y)
    bx = [float(v) for v in beta_x]
    by = [float(v) for v in beta_y]
    var = ivw_variance(bx, se_x, by, se_y, weights, theta)
    ratios = ratio_estimates(bx, by)
    w = [1.0 / v for v in var]
    est = (sum(ratios[j] * w[j] for j in range(L))
           / sum(w))
    se_fixed = math.sqrt(1.0 / sum(w))
    het = heterogeneity(ratios, var, est)
    reg = weighted_regression(bx, by, [1.0 / (float(se_y[j]) ** 2)
                                       for j in range(L)])
    if model == "fixed":
        se = se_fixed
        phi = 1.0
    elif model == "multiplicative":
        phi = max(math.sqrt(het["Q"] / het["df"]), 1.0) \
            if het["df"] > 0 else 1.0
        se = se_fixed * phi
    else:
        w2 = [1.0 / (var[j] + het["tau2"]) for j in range(L)]
        est = sum(ratios[j] * w2[j] for j in range(L)) / sum(w2)
        se = math.sqrt(1.0 / sum(w2))
        phi = 1.0
    z = est / se if se > 0.0 else float("inf")
    return RichResult(payload={
        "estimate": est, "se": se, "z": z,
        "p_value": math.erfc(abs(z) / math.sqrt(2.0)),
        "ci": (est - 1.96 * se, est + 1.96 * se),
        "ratio_estimates": ratios, "variances": var,
        "weights_used": weights, "model": model,
        "phi_multiplicative": phi, "tau2": het["tau2"],
        "Q": het["Q"], "df": het["df"], "I2": het["I2"],
        "se_fixed": se_fixed,
        "regression_estimate": reg["estimate"],
        "regression_se_fixed": reg["se_fixed"],
        "n_variants": L,
        "method": "inverse-variance weighted MR (%s model, %s "
                  "weights); Burgess & Bowden (2015) Sec. 2"
                  % (model, weights),
    })


def cheatsheet():
    return ("mtr2sx: IVW = weighted mean of the per-variant ratio "
            "estimates beta_Y/beta_X, identically a weighted "
            "regression of beta_Y on beta_X through the origin with "
            "sigma_Y^-2 weights, identically 2SLS for uncorrelated "
            "variants. First-order weights over-reject under sample "
            "overlap -- use the second-order delta weights with "
            "theta. Fixed and multiplicative share a point estimate "
            "and differ by phi_M (floored at 1); the additive "
            "DerSimonian-Laird model moves the estimate too. "
            "Multiplicative is the default, as the paper recommends.")


# compact alias per ledger/NAMING.md
mendelian_randomization_ivw = ivw
