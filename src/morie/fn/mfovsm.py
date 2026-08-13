# morie.fn -- function file (rootcoder007/morie)
r"""Marginal structural model with effect modification by a baseline
feature.

Robins & Hernan (2009) Sec. 4 gives the marginal structural model for a
time-varying treatment *conditional on a baseline covariate* V, which
Hernan & Robins (2020) Sec. 12.5 states as

.. math:: E\left[Y^{\bar a} \mid V\right] = \beta_0 + \beta_1 \bar a
          + \beta_2 V + \beta_3 \bar a V,

so that :math:`\beta_3` is the effect modification: how much the causal
effect of exposure differs across levels of V. The distinction the
sources insist on, and the reason this is not the same as adding V to
an outcome regression, is that V must be a **baseline** feature. It is
measured before treatment, so conditioning on it cannot induce
collider bias, and the model still refers to counterfactual means
rather than to observed conditional means.

Two things follow, and both are implemented rather than assumed.

**V belongs in the weight numerator.** Sec. 12.5: when the MSM
conditions on V, the stabilized weights use
:math:`f(A \mid V) / f(A \mid L)` rather than
:math:`f(A) / f(A \mid L)`. Putting V in the numerator makes the
weights smaller and the estimate more precise, and leaves the estimand
unchanged. ``v_in_numerator=False`` is available for comparison and the
anchor checks that the two agree on the effect while differing on the
variance -- which is exactly the claim.

**V must not be post-treatment.** There is no way to verify that from
the data, so it is stated in the docstring and the argument is named
`feature` rather than `covariate` to keep it distinct from the
time-varying `H` that goes in the weight denominator.

References
----------
Robins, J. M. & Hernan, M. A. (2009) "Estimation of the causal effects
of time-varying exposures", in Fitzmaurice, G., Davidian, M.,
Verbeke, G. & Molenberghs, G. (eds.), *Longitudinal Data Analysis*,
Chapman & Hall/CRC Handbooks of Modern Statistical Methods, 553-599,
doi:10.1201/9781420011579.ch23.

Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*,
Chapman & Hall/CRC, Sec. 12.5 (effect modification and marginal
structural models) and Sec. 21.2 (the time-varying weights).
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["mfo_vsm"]


def mfo_vsm(y, feature, A, H, v_in_numerator=True, contrast="cumulative",
            trim=None):
    r"""MSM with effect modification by a baseline feature V.

    Parameters
    ----------
    y : array-like
        Outcome.
    feature : array-like
        The baseline effect modifier V. Must be measured before
        treatment; conditioning on a post-treatment feature makes the
        model something other than a marginal structural model and
        nothing here can detect that.
    A : list of array-like
        Treatment at each time point.
    H : list of array-like
        Time-varying covariates for the weight denominator.
    v_in_numerator : bool
        Use f(A|V)/f(A|L) rather than f(A)/f(A|L), per Sec. 12.5.

    Returns
    -------
    RichResult
        ``estimate`` is beta_3, the interaction -- the effect
        modification itself. ``main_effect`` is beta_1.

    Examples
    --------
    Effect modification by a binary baseline feature::

        r = mfo_vsm(y, V, [A0, A1], [L0, L1])
        r["estimate"]     # beta_3
    """
    A_hist = _hist(A)
    L_hist = _hist(H, allow_none=True)
    if len(L_hist) != len(A_hist):
        raise ValueError("mfo_vsm: %d treatment times but %d covariate "
                         "blocks" % (len(A_hist), len(L_hist)))
    yv = k.vec(y)
    vv = k.vec(feature)
    n = len(yv)
    if len(vv) != n:
        raise ValueError("mfo_vsm: outcome has %d rows but the feature "
                         "has %d" % (n, len(vv)))

    # Sec. 21.2's product, with V added to the numerator model when
    # asked -- Sec. 12.5's refinement for a V-conditional MSM.
    w = [1.0] * n
    past = []
    lbar = []
    for kk in range(len(A_hist)):
        ak = k.vec(A_hist[kk])
        if L_hist[kk] is not None:
            block = k.mat(L_hist[kk])
            for c in range(len(block[0])):
                lbar.append([row[c] for row in block])
        den = [[c[i] for c in lbar + past] for i in range(n)] \
            if (lbar or past) else None
        num_cols = ([list(vv)] if v_in_numerator else []) + past
        num = [[c[i] for c in num_cols] for i in range(n)] \
            if num_cols else None
        wk, _ = k.ip_weights(ak, den, num, kind="binary", stabilize=True)
        for i in range(n):
            w[i] *= wk[i]
        past = past + [list(ak)]
    if trim is not None:
        q = float(trim)
        if not 0.5 < q < 1.0:
            raise ValueError("mfo_vsm: trim must be in (0.5, 1)")
        hi, lo = k.quantile7(w, q), k.quantile7(w, 1.0 - q)
        w = [min(max(v, lo), hi) for v in w]

    cum = [sum(k.vec(a)[i] for a in A_hist) for i in range(n)]
    if contrast == "cumulative":
        e = cum
    elif contrast == "final":
        e = list(k.vec(A_hist[-1]))
    elif contrast == "everexposed":
        e = [1.0 if v > 0.0 else 0.0 for v in cum]
    else:
        raise ValueError("mfo_vsm: contrast must be 'cumulative', "
                         "'final' or 'everexposed', got %r" % (contrast,))

    X = [[e[i], vv[i], e[i] * vv[i]] for i in range(n)]
    fit = k.wls(X, yv, w)
    s1, s2 = sum(w), sum(v * v for v in w)
    return RichResult(payload={
        "estimate": fit["coef"][3],          # beta_3, the modification
        "se": fit["se"][3],
        "main_effect": fit["coef"][1],       # beta_1
        "main_effect_se": fit["se"][1],
        "feature_effect": fit["coef"][2],    # beta_2
        "intercept": fit["coef"][0],
        "coef": fit["coef"], "vcov": fit["vcov"],
        "weights": w, "mean_weight": s1 / n, "max_weight": max(w),
        "effective_sample_size": (s1 * s1 / s2) if s2 else 0.0,
        "exposure": e, "v_in_numerator": bool(v_in_numerator),
        "n": n, "n_times": len(A_hist), "contrast": contrast,
        "method": "V-conditional marginal structural model, Robins & "
                  "Hernan (2009); Hernan & Robins (2020) Sec. 12.5",
    })


def _hist(obj, allow_none=False):
    if obj is None:
        return [None] if allow_none else []
    if isinstance(obj, (list, tuple)) and obj and (
            isinstance(obj[0], (list, tuple)) or obj[0] is None
            or hasattr(obj[0], "shape")):
        return list(obj)
    return [obj]


def cheatsheet():
    return ("mfovsm: V-conditional MSM E[Y^abar|V] = b0 + b1 abar + "
            "b2 V + b3 abar V (Robins-Hernan 2009; H&R Sec.12.5). "
            "estimate = b3, the effect modification. V goes in the "
            "weight NUMERATOR: f(A|V)/f(A|L).")


# compact alias per ledger/NAMING.md
mfovsm_ = mfo_vsm
