# morie.fn -- function file (rootcoder007/morie)
r"""Regression kink design: an effect identified by a change in slope.

A regression discontinuity uses a **jump** in treatment at a
threshold. Many policies have no jump: unemployment benefits rise with
prior earnings up to a cap, then flatten. Nothing discontinuous
happens at the cap -- the *level* of benefits is continuous -- but the
**slope** changes sharply. A regression kink design identifies the
causal effect from that change in slope.

**The estimand.** With :math:`B` the policy variable, :math:`V` the
assignment variable and :math:`k` the kink point, the sharp RKD
estimand is the ratio of the change in the slope of the outcome to the
change in the slope of the policy:

.. math:: \tau_{RKD} =
          \frac{\lim_{v \downarrow k}
                \frac{dE[Y \mid V = v]}{dv}
              - \lim_{v \uparrow k}
                \frac{dE[Y \mid V = v]}{dv}}
               {\lim_{v \downarrow k} \frac{dB(v)}{dv}
              - \lim_{v \uparrow k} \frac{dB(v)}{dv}}.

Numerator and denominator are both *kinks*, not jumps. The
denominator is usually known exactly from the policy rule, which is
what makes the design attractive: the first stage is legislation
rather than an estimate.

**Why it can fail, and the test that detects it.** Identification
needs the density of the assignment variable to be **smooth** at the
kink. If agents can precisely manipulate :math:`V` -- reporting
earnings just below a cap -- the density itself kinks there, and so
does the composition of who lands on each side. The design then
attributes a compositional change to the policy. So the density is
tested for a kink, and covariates are tested for kinks too: a
covariate that should be unaffected by the policy must not bend at
:math:`k`.

**Fitting on each side separately, and why the bandwidth matters more
here.** Local linear regression estimates a *level* and needs the
first derivative to behave; RKD estimates a *slope*, one derivative
further up, so it is more sensitive to both bandwidth and polynomial
order. ``order`` is exposed, and the anchor measures that a linear fit
on a genuinely quadratic surface is biased while a quadratic fit is
not -- rather than asserting that some default is safe.

**Fuzzy RKD.** When the policy rule is not followed exactly, the
denominator is estimated from data as the kink in observed treatment,
and the estimator becomes a ratio of two estimated kinks -- the kink
analogue of fuzzy RD. ``fuzzy=True`` does that and reports both
kinks, since a small denominator is the failure mode to watch.

References
----------
Card, D., Lee, D. S., Pei, Z. & Weber, A. "Nonlinear Policy Rules and
the Identification and Estimation of Causal Effects in a Generalized
Regression Kink Design", NBER Working Paper 18564, November 2012;
published as "Inference on Causal Effects in a Generalized Regression
Kink Design", *Econometrica* 83(6), 2453-2483 (2015),
doi:10.3982/ECTA11224. The identification of causal effects from a
change in the slope of the policy rule, the smooth-density condition
and its testable implications, and estimation by local polynomial
regression on each side of the kink.

Imbens, G. & Kalyanaraman, K. (2012) "Optimal Bandwidth Choice for
the Regression Discontinuity Estimator", *The Review of Economic
Studies* 79(3), 933-959, doi:10.1093/restud/rdr043. The bandwidth
problem, which RKD inherits in a sharper form because a derivative is
being estimated.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["local_polynomial_slope", "rkd_estimate",
           "density_kink_test", "covariate_kink_test"]

_EPS = 1e-12


def _side_fit(v, y, k_pt, bandwidth, order, side, kernel):
    """Local polynomial on one side; returns the fitted derivative
    at the kink."""
    rows, ys, w = [], [], []
    for i in range(len(v)):
        d = float(v[i]) - float(k_pt)
        if side == "right" and d < 0.0:
            continue
        if side == "left" and d > 0.0:
            continue
        if abs(d) > float(bandwidth):
            continue
        u = abs(d) / float(bandwidth)
        kw = (1.0 - u) if kernel == "triangular" else 1.0
        if kw <= 0.0:
            continue
        rows.append([d ** p for p in range(1, int(order) + 1)])
        ys.append(float(y[i]))
        w.append(kw)
    if len(rows) < int(order) + 1:
        raise ValueError("bnskt2: too few observations on the %s of "
                         "the kink within the bandwidth (%d for order "
                         "%d)" % (side, len(rows), order))
    fit = k.wls(rows, ys, w)
    # coefficient on d is the derivative at the kink
    return {"slope": fit["coef"][1], "coef": fit["coef"],
            "n": len(rows)}


def local_polynomial_slope(v, y, kink, bandwidth, order=2,
                           side="right", kernel="triangular"):
    r"""The one-sided derivative of :math:`E[Y \mid V]` at the kink."""
    if side not in ("left", "right"):
        raise ValueError("bnskt2: side must be left or right, got %r"
                         % (side,))
    if kernel not in ("triangular", "uniform"):
        raise ValueError("bnskt2: kernel must be triangular or "
                         "uniform, got %r" % (kernel,))
    if int(order) < 1:
        raise ValueError("bnskt2: the polynomial order must be at "
                         "least 1")
    if float(bandwidth) <= 0.0:
        raise ValueError("bnskt2: the bandwidth must be positive")
    return _side_fit(k.vec(v), k.vec(y), kink, bandwidth, order, side,
                     kernel)


def rkd_estimate(V, Y, kink, bandwidth, order=2, kernel="triangular",
                 policy_slope_change=None, B=None, fuzzy=False):
    r"""The RKD estimand: a ratio of kinks.

    Supply ``policy_slope_change`` when the policy rule is known --
    the usual sharp case, where the denominator is legislation rather
    than an estimate. Set ``fuzzy=True`` with ``B`` to estimate the
    denominator from observed treatment instead.
    """
    v = [float(x) for x in k.vec(V)]
    y = [float(x) for x in k.vec(Y)]
    if len(v) != len(y):
        raise ValueError("bnskt2: V and Y must agree in length "
                         "(%d, %d)" % (len(v), len(y)))
    r = _side_fit(v, y, kink, bandwidth, order, "right", kernel)
    l = _side_fit(v, y, kink, bandwidth, order, "left", kernel)
    num = r["slope"] - l["slope"]
    if fuzzy:
        if B is None:
            raise ValueError("bnskt2: fuzzy RKD needs the observed "
                             "treatment B")
        b = [float(x) for x in k.vec(B)]
        if len(b) != len(v):
            raise ValueError("bnskt2: B has %d entries for %d "
                             "observations" % (len(b), len(v)))
        rb = _side_fit(v, b, kink, bandwidth, order, "right", kernel)
        lb = _side_fit(v, b, kink, bandwidth, order, "left", kernel)
        den = rb["slope"] - lb["slope"]
        den_src = "estimated from observed treatment"
    else:
        if policy_slope_change is None:
            raise ValueError("bnskt2: sharp RKD needs "
                             "policy_slope_change, the known change "
                             "in the slope of the policy rule")
        den = float(policy_slope_change)
        den_src = "known policy rule"
    if abs(den) <= _EPS:
        raise ValueError("bnskt2: the change in the policy slope is "
                         "zero (%.3g) -- there is no kink to "
                         "identify from" % den)
    return RichResult(payload={
        "estimate": num / den, "tau": num / den,
        "outcome_kink": num, "policy_kink": den,
        "slope_right": r["slope"], "slope_left": l["slope"],
        "n_right": r["n"], "n_left": l["n"],
        "bandwidth": float(bandwidth), "order": int(order),
        "kernel": kernel, "fuzzy": bool(fuzzy),
        "denominator_source": den_src,
        "method": "regression kink design; Card, Lee, Pei & Weber "
                  "(NBER WP 18564 / Econometrica 2015)",
        "requires": "the density of V must be smooth at the kink -- "
                    "test it",
    })


def density_kink_test(V, kink, bandwidth, n_bins=20, order=1):
    r"""Does the density of the assignment variable itself kink?

    Precise manipulation of :math:`V` bends its density at
    :math:`k`, which invalidates the design. Bins the data, fits the
    log-density slope on each side, and reports the change.
    """
    v = [float(x) for x in k.vec(V)]
    kp, bw = float(kink), float(bandwidth)
    inside = [x for x in v if abs(x - kp) <= bw]
    if len(inside) < 4 * int(n_bins):
        raise ValueError("bnskt2: too few observations within the "
                         "bandwidth for %d bins" % n_bins)
    edges = [kp - bw + 2.0 * bw * i / n_bins
             for i in range(int(n_bins) + 1)]
    ctr, dens = [], []
    for b in range(int(n_bins)):
        c = 0.5 * (edges[b] + edges[b + 1])
        cnt = sum(1 for x in inside if edges[b] <= x < edges[b + 1])
        ctr.append(c)
        dens.append(cnt / float(len(inside)))
    right = _side_fit(ctr, dens, kp, bw, order, "right", "uniform")
    left = _side_fit(ctr, dens, kp, bw, order, "left", "uniform")
    change = right["slope"] - left["slope"]
    scale = max(sum(dens) / len(dens), _EPS)
    return {"slope_change": change, "relative": change / scale,
            "slope_right": right["slope"],
            "slope_left": left["slope"],
            "n_inside": len(inside), "n_bins": int(n_bins),
            "smooth": abs(change / scale) < 1.0,
            "interpretation": "a kink in the DENSITY suggests precise "
                              "manipulation of the assignment "
                              "variable, which invalidates the design"}


def covariate_kink_test(V, Z, kink, bandwidth, order=2,
                        kernel="triangular"):
    r"""A covariate that the policy cannot affect must not kink."""
    r = _side_fit(k.vec(V), k.vec(Z), kink, bandwidth, order, "right",
                  kernel)
    l = _side_fit(k.vec(V), k.vec(Z), kink, bandwidth, order, "left",
                  kernel)
    return {"slope_change": r["slope"] - l["slope"],
            "slope_right": r["slope"], "slope_left": l["slope"],
            "n_right": r["n"], "n_left": l["n"],
            "interpretation": "a kink here is evidence the design is "
                              "picking up composition rather than the "
                              "policy"}


def cheatsheet():
    return ("bnskt2: regression KINK design. RD uses a JUMP in "
            "treatment; RKD uses a change in SLOPE -- benefits rising "
            "with earnings up to a cap, then flat. tau = (change in "
            "the slope of E[Y|V]) / (change in the slope of the "
            "policy). The denominator is usually KNOWN from "
            "legislation, so the first stage is not estimated. Needs "
            "the density of V SMOOTH at the kink -- precise "
            "manipulation bends it, and then composition is mistaken "
            "for policy. More bandwidth-sensitive than RD because a "
            "DERIVATIVE is being estimated.")


# compact alias per ledger/NAMING.md
kinktreatmentbound = rkd_estimate

# public names resolved by fn/_lazy_map.json
bound_kink_te = rkd_estimate
boundkinkte = rkd_estimate
