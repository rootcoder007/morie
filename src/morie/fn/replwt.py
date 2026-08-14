# morie.fn -- function file (rootcoder007/morie)
r"""Replicate weights: variance without a variance formula.

**The problem.** A survey estimate is a function of weighted data
under a design with strata, clusters and unequal weights. Writing
down its sampling variance analytically means differentiating that
function -- feasible for a total, painful for a ratio, unpleasant for
a quantile or a regression coefficient. Replication sidesteps it:
build :math:`R` sets of weights that mimic resampling the design,
recompute the *same* estimator under each, and read the variance off
the spread.

.. math:: \hat{V}(\hat\theta) = c \sum_{r=1}^{R}
          (\hat\theta_{(r)} - \hat\theta)^2

The estimator is a black box. Only :math:`c` and the weights change
between methods.

**Jackknife (JK1, JKn).** Drop one PSU and inflate the survivors in
its stratum. ``jk1`` treats the sample as unstratified and uses
:math:`c = (n-1)/n`; ``jkn`` drops one PSU per stratum in turn with
:math:`c_h = (n_h - 1)/n_h`. For the mean of a simple random sample
this reproduces :math:`s^2/n` exactly, which is the anchor.

**BRR.** With exactly two PSUs per stratum the jackknife needs
:math:`2H` replicates; balanced repeated replication needs the next
multiple of four. Each replicate keeps one PSU per stratum at double
weight and drops the other, and the pattern of which is taken comes
from a Hadamard matrix, so the selections are mutually orthogonal.
With a full orthogonal set the estimator is not an approximation: for
a linear statistic it equals the textbook stratified variance
:math:`\sum_h (y_{h1} - y_{h2})^2 / 4` **exactly**. The anchor checks
that equality rather than a tolerance.

**Fay's variant.** Dropping a PSU entirely makes subdomain estimates
undefined whenever a domain lives in the dropped half. Fay's
modification keeps the dropped half at :math:`\rho` and the retained
half at :math:`2 - \rho`, with :math:`c = 1/(R(1-\rho)^2)`. The two
factors still sum to two, so the total weight is preserved, and at
:math:`\rho = 0` this *is* BRR -- an identity, and one the anchor
checks rather than a limit it takes on trust.

**Rao-Wu bootstrap.** Resample :math:`n_h - 1` PSUs with replacement
per stratum and rescale so the weights stay design-unbiased. The one
method here that needs a random number stream, so the seed is an
argument and is reported.

References
----------
Wolter, K. M. (2007) *Introduction to Variance Estimation*, 2nd
edition, Springer, ISBN 978-0-387-32917-8,
doi:10.1007/978-0-387-35099-8. Ch. 2 (random groups), Ch. 3 (balanced
half-samples, the Hadamard construction, Fay's modification), Ch. 4
(the jackknife, JK1 and JKn multipliers) and Ch. 5 (the bootstrap);
the exact equivalence of full-orthogonal BRR to the stratified
variance estimator for linear statistics.

McCarthy, P. J. (1969) "Pseudo-replication: half samples", *Review of
the International Statistical Institute* 37(3), 239-264,
doi:10.2307/1402116, for balanced half-sample replication itself.

Rao, J. N. K. & Wu, C. F. J. (1988) "Resampling inference with complex
survey data", *Journal of the American Statistical Association*
83(401), 231-241, doi:10.1080/01621459.1988.10478591, for the
rescaling bootstrap.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["design", "hadamard", "jackknife_weights", "brr_weights",
           "bootstrap_weights", "replicate_variance",
           "replicate_weights", "METHODS"]

METHODS = ("jk1", "jkn", "brr", "fay", "bootstrap")


def design(weights, strata=None, psu=None):
    r"""A survey design: sampling weights, strata and PSU labels."""
    w = [float(x) for x in weights]
    n = len(w)
    if n < 2:
        raise ValueError("replwt: a design needs at least two units")
    if any(x <= 0.0 for x in w):
        raise ValueError("replwt: sampling weights must be positive")
    h = ["1"] * n if strata is None else [str(x) for x in strata]
    p = [str(i) for i in range(n)] if psu is None else [str(x)
                                                        for x in psu]
    if len(h) != n or len(p) != n:
        raise ValueError("replwt: strata and psu must have one entry "
                         "per unit (%d)" % n)
    order, groups = [], {}
    for i in range(n):
        key = (h[i], p[i])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    by_stratum = {}
    for (hh, pp) in order:
        by_stratum.setdefault(hh, []).append((hh, pp))
    for hh, ps in by_stratum.items():
        if len(ps) < 2:
            raise ValueError("replwt: stratum %r has a single PSU, so "
                             "its contribution to the variance is not "
                             "estimable" % hh)
    return {"weights": w, "strata": h, "psu": p, "n": n,
            "psu_order": order, "psu_units": groups,
            "stratum_psus": by_stratum,
            "stratum_order": [hh for hh in
                              dict.fromkeys(h)]}


def hadamard(order):
    r"""A Sylvester Hadamard matrix of +-1, for orders 1, 2, 4, 8, ...

    Rows are mutually orthogonal, which is what makes the half-sample
    selections balanced.
    """
    k = int(order)
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError("replwt: this construction gives Hadamard "
                         "matrices of order a power of two; %d is "
                         "not one" % k)
    H = [[1]]
    while len(H) < k:
        H = ([r + r for r in H]
             + [r + [-v for v in r] for r in H])
    return H


def _psu_totals(d, values):
    """Weighted total per PSU, in psu_order."""
    out = []
    for key in d["psu_order"]:
        idx = d["psu_units"][key]
        out.append(sum(d["weights"][i] * values[i] for i in idx))
    return out


def jackknife_weights(d, method="jkn"):
    r"""One replicate per PSU: drop it, inflate its stratum."""
    if method not in ("jk1", "jkn"):
        raise ValueError("replwt: jackknife method must be jk1 or "
                         "jkn, got %r" % method)
    reps, drop = [], []
    if method == "jk1":
        m = len(d["psu_order"])
        for key in d["psu_order"]:
            w = list(d["weights"])
            for i in d["psu_units"][key]:
                w[i] = 0.0
            f = m / float(m - 1)
            for i in range(d["n"]):
                if i not in d["psu_units"][key]:
                    w[i] *= f
            reps.append(w)
            drop.append(key)
        return {"weights": reps, "dropped": drop,
                "scale": [(m - 1) / float(m)] * m, "method": "jk1"}
    scale = []
    for hh in d["stratum_order"]:
        ps = d["stratum_psus"][hh]
        nh = len(ps)
        for key in ps:
            w = list(d["weights"])
            for i in d["psu_units"][key]:
                w[i] = 0.0
            for other in ps:
                if other == key:
                    continue
                for i in d["psu_units"][other]:
                    w[i] *= nh / float(nh - 1)
            reps.append(w)
            drop.append(key)
            scale.append((nh - 1) / float(nh))
    return {"weights": reps, "dropped": drop, "scale": scale,
            "method": "jkn"}


def brr_weights(d, fay=0.0):
    r"""Half-sample replicates from a Hadamard matrix.

    ``fay`` is :math:`\rho`: 0 gives classical BRR (factors 2 and
    0), a positive value keeps the dropped half at :math:`\rho` and
    the retained half at :math:`2 - \rho`.
    """
    rho = float(fay)
    if not 0.0 <= rho < 1.0:
        raise ValueError("replwt: Fay's rho must lie in [0, 1), got "
                         "%g" % rho)
    strata = d["stratum_order"]
    for hh in strata:
        if len(d["stratum_psus"][hh]) != 2:
            raise ValueError("replwt: BRR needs exactly two PSUs per "
                             "stratum; stratum %r has %d"
                             % (hh, len(d["stratum_psus"][hh])))
    H = len(strata)
    R = 1
    while R < H + 1:
        R *= 2
    M = hadamard(R)
    reps = []
    for r in range(R):
        w = list(d["weights"])
        for j, hh in enumerate(strata):
            a, b = d["stratum_psus"][hh]
            keep, drop_ = (a, b) if M[r][j + 1] > 0 else (b, a)
            # Fay: the kept half goes to 2 - rho and the dropped
            # half to rho, so rho = 0 IS classical BRR (2 and 0) and
            # the two factors always sum to 2.
            for i in d["psu_units"][keep]:
                w[i] *= 2.0 - rho
            for i in d["psu_units"][drop_]:
                w[i] *= rho
        reps.append(w)
    c = 1.0 / (R * (1.0 - rho) ** 2)
    return {"weights": reps, "scale": [c] * R, "n_replicates": R,
            "hadamard_order": R, "fay": rho,
            "method": "fay" if rho else "brr"}


def bootstrap_weights(d, R=200, seed=1):
    r"""Rao-Wu rescaling bootstrap: resample n_h - 1 PSUs per stratum."""
    if int(R) < 2:
        raise ValueError("replwt: need at least two bootstrap "
                         "replicates")
    rng = np.random.default_rng(int(seed))
    reps = []
    for _ in range(int(R)):
        w = list(d["weights"])
        for hh in d["stratum_order"]:
            ps = d["stratum_psus"][hh]
            nh = len(ps)
            mh = nh - 1
            count = {k: 0 for k in ps}
            for _j in range(mh):
                count[ps[int(rng.random() * nh) % nh]] += 1
            root = math.sqrt(mh / float(nh - 1))
            for k in ps:
                f = (1.0 - root
                     + root * (nh / float(mh)) * count[k])
                for i in d["psu_units"][k]:
                    w[i] *= f
        reps.append(w)
    return {"weights": reps, "scale": [1.0 / int(R)] * int(R),
            "n_replicates": int(R), "seed": int(seed),
            "method": "bootstrap"}


def replicate_variance(estimator, d, rep, values=None):
    r"""Run ``estimator`` under every replicate and read off the
    variance.

    ``estimator`` takes a weight vector (and ``values`` if given) and
    returns a number.
    """
    def call(w):
        return (float(estimator(w)) if values is None
                else float(estimator(w, values)))

    theta = call(d["weights"])
    reps = [call(w) for w in rep["weights"]]
    v = sum(s * (t - theta) ** 2
            for s, t in zip(rep["scale"], reps))
    return RichResult(payload={
        "estimate": theta, "theta": theta, "variance": v,
        "std_error": math.sqrt(v) if v >= 0 else float("nan"),
        "replicates": reps, "n_replicates": len(reps),
        "method": rep["method"],
    })


def replicate_weights(d, method="jkn", R=200, fay=0.0, seed=1):
    r"""Entry point: build a replicate-weight set for a design."""
    if method not in METHODS:
        raise ValueError("replwt: method must be one of %s, got %r"
                         % (", ".join(METHODS), method))
    if method in ("jk1", "jkn"):
        rep = jackknife_weights(d, method)
    elif method == "brr":
        rep = brr_weights(d, 0.0)
    elif method == "fay":
        if not fay:
            raise ValueError("replwt: Fay's method needs a non-zero "
                             "rho; use method='brr' for rho = 0")
        rep = brr_weights(d, fay)
    else:
        rep = bootstrap_weights(d, R, seed)
    return RichResult(payload={
        "estimate": rep["weights"], "weights": rep["weights"],
        "scale": rep["scale"], "n_replicates": len(rep["weights"]),
        "method": rep["method"],
        "dropped": rep.get("dropped"),
        "hadamard_order": rep.get("hadamard_order"),
        "fay": rep.get("fay"), "seed": rep.get("seed"),
    })
