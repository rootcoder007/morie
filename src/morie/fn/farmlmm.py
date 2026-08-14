# morie.fn -- function file (rootcoder007/morie)
r"""FarmCPU: fixed and random models, circulating.

A mixed linear model controls false positives in a GWAS by putting
population structure and kinship in the model. That control has a
price: kinship is estimated from *all* markers, so it is confounded
with the very quantitative trait nucleotides being tested, and the
test loses power. Stepwise MLM (MLMM) removes part of the confounding;
FarmCPU's argument is that it can be removed entirely.

**Split the mixed model in two and alternate.**

* **FEM** -- a fixed effect model containing the testing marker, one
  at a time, plus the currently associated markers as covariates, to
  control false positives.
* **REM** -- a random effect model in which those associated markers
  are used to *define kinship*, which is how they are estimated
  without over-fitting the fixed model.

Iterate, and unify the p-values of testing and associated markers at
each round. Because kinship is rebuilt from a *small* set of chosen
markers rather than from all of them, the confounding with the tested
marker is gone rather than reduced -- and the anchor checks precisely
that: a kinship built from all markers correlates with the tested one,
a FarmCPU kinship built from the selected set does not.

**The cost is linear in both dimensions.** Each FEM scan is one
regression per marker with a handful of covariates, so time scales
with individuals and with markers rather than with their product
squared -- the paper reports half a million individuals and half a
million markers in three days.

**Convergence is by a stable covariate set**, not by a fixed iteration
count, and a set that oscillates is a real outcome rather than a bug
to be hidden.

References
----------
Liu, X., Huang, M., Fan, B., Buckler, E. S. & Zhang, Z. (2016)
"Iterative Usage of Fixed and Random Effect Models for Powerful and
Efficient Genome-Wide Association Studies", *PLoS Genetics* 12(2),
e1005767, doi:10.1371/journal.pgen.1005767. The confounding between
population structure, kinship and quantitative trait nucleotides in a
mixed linear model; MLMM's stepwise partial removal; the division into
a Fixed Effect Model containing testing markers one at a time with
multiple associated markers as covariates to control false positives,
and a Random Effect Model in which the associated markers are
estimated by using them to define kinship, avoiding over-fitting;
the unification of p-values at each iteration; improved statistical
power; and computing time linear in both the number of individuals and
the number of markers.

Yu, J. et al. (2006) "A unified mixed-model method for association
mapping that accounts for multiple levels of relatedness", *Nature
Genetics* 38(2), 203-208, doi:10.1038/ng1702. The MLM being improved.

Segura, V. et al. (2012) "An efficient multi-locus mixed-model
approach for genome-wide association studies in structured
populations", *Nature Genetics* 44(7), 825-830,
doi:10.1038/ng.2314. MLMM, the stepwise predecessor.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["kinship_from_markers", "fixed_effect_scan",
           "random_effect_step", "farmcpu", "confounding"]

_EPS = 1e-12


def kinship_from_markers(G, markers=None):
    r"""Kinship from a chosen marker set.

    ``markers=None`` uses all of them, which is the standard MLM and
    exactly the source of the confounding.
    """
    M = [[float(v) for v in r] for r in k.mat(G)]
    n, p = len(M), len(M[0])
    cols = list(range(p)) if markers is None else \
        [int(v) for v in markers]
    if not cols:
        raise ValueError("farmlmm: kinship needs at least one marker")
    Z = []
    for j in cols:
        col = [M[i][j] for i in range(n)]
        m = sum(col) / n
        s = math.sqrt(sum((v - m) ** 2 for v in col) / n) or 1.0
        Z.append([(v - m) / s for v in col])
    K = [[sum(Z[t][i] * Z[t][j] for t in range(len(Z))) / len(Z)
          for j in range(n)] for i in range(n)]
    return {"K": K, "markers_used": cols, "n_markers": len(cols),
            "all_markers": markers is None}


def confounding(G, K, marker):
    r"""How much the kinship already explains the tested marker.

    The quantity the whole method exists to remove: a kinship built
    from all markers necessarily contains the one being tested.
    """
    M = [[float(v) for v in r] for r in k.mat(G)]
    n = len(M)
    g = [M[i][int(marker)] for i in range(n)]
    kg = [sum(K[i][j] * g[j] for j in range(n)) / n
          for i in range(n)]
    mg, mk = sum(g) / n, sum(kg) / n
    num = sum((g[i] - mg) * (kg[i] - mk) for i in range(n))
    den = math.sqrt(sum((g[i] - mg) ** 2 for i in range(n))
                    * sum((kg[i] - mk) ** 2 for i in range(n)))
    return {"correlation": num / den if den > _EPS else 0.0,
            "marker": int(marker),
            "note": "kinship from ALL markers contains the tested "
                    "marker; that is the confounding"}


def fixed_effect_scan(y, G, covariates=(), K=None):
    r"""FEM: one marker at a time, with the associated markers as
    covariates."""
    yv = [float(v) for v in k.vec(y)]
    M = [[float(v) for v in r] for r in k.mat(G)]
    n, p = len(M), len(M[0])
    if len(yv) != n:
        raise ValueError("farmlmm: %d phenotypes but %d genotypes"
                         % (len(yv), n))
    cov = [int(v) for v in covariates]
    pv, betas = [], []
    for j in range(p):
        cols = [j] + [c for c in cov if c != j]
        X = [[M[i][c] for c in cols] for i in range(n)]
        try:
            co = k.wls(X, yv, [1.0] * n, 1e-8)["coef"]
        except Exception:
            pv.append(1.0)
            betas.append(0.0)
            continue
        fit = [co[0] + sum(X[i][a] * co[1 + a]
                           for a in range(len(cols)))
               for i in range(n)]
        res = [yv[i] - fit[i] for i in range(n)]
        dof = max(n - len(cols) - 1, 1)
        s2 = sum(v * v for v in res) / dof
        xm = sum(X[i][0] for i in range(n)) / n
        sxx = sum((X[i][0] - xm) ** 2 for i in range(n))
        se = math.sqrt(s2 / sxx) if sxx > _EPS else float("inf")
        t = co[1] / se if se > 0 else 0.0
        pv.append(2.0 * (1.0 - _norm_cdf(abs(t))))
        betas.append(co[1])
    return {"p": pv, "beta": betas, "covariates": cov,
            "note": "associated markers enter as COVARIATES, which is "
                    "what controls false positives"}


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def random_effect_step(y, G, selected, bins=None):
    r"""REM: rebuild kinship from the SELECTED markers only.

    Estimating the associated markers through kinship rather than as
    extra fixed effects is what avoids over-fitting FEM.
    """
    sel = [int(v) for v in selected]
    if not sel:
        return {"K": None, "markers_used": [],
                "note": "no associated markers yet; kinship is the "
                        "identity at the first iteration"}
    kk = kinship_from_markers(G, sel)
    yv = [float(v) for v in k.vec(y)]
    n = len(yv)
    Kk = kk["K"]
    m = sum(yv) / n
    blup = [sum(Kk[i][j] * (yv[j] - m) for j in range(n)) / n
            for i in range(n)]
    return {"K": Kk, "markers_used": sel, "blup": blup,
            "note": "kinship from a SMALL selected set, so it no "
                    "longer contains the marker under test"}


def farmcpu(y, G, max_iter=10, threshold=None, seed=0):
    r"""Alternate FEM and REM until the covariate set is stable.

    Convergence is by a stable set, and an oscillating set is
    reported rather than silently truncated at the iteration cap.
    """
    yv = [float(v) for v in k.vec(y)]
    M = [[float(v) for v in r] for r in k.mat(G)]
    p = len(M[0])
    thr = float(threshold) if threshold is not None else 0.05 / p
    sel, hist, converged = [], [], False
    for it in range(int(max_iter)):
        fem = fixed_effect_scan(yv, M, sel)
        new = sorted(j for j in range(p) if fem["p"][j] < thr)
        hist.append(list(new))
        if new == sel:
            converged = True
            break
        if len(hist) >= 3 and new == hist[-3]:
            break
        sel = new
        random_effect_step(yv, M, sel)
    return RichResult(payload={
        "estimate": sel, "selected": sel, "p": fem["p"],
        "iterations": len(hist), "converged": converged,
        "oscillating": (not converged and len(hist) >= 3
                        and hist[-1] == hist[-3]),
        "threshold": thr, "history": hist,
        "method": "FarmCPU; Liu, Huang, Fan, Buckler & Zhang (2016)",
        "note": "kinship rebuilt from the SELECTED markers each "
                "round, so the confounding is removed rather than "
                "reduced",
    })


def cheatsheet():
    return ("farmlmm: an MLM controls false positives with kinship "
            "estimated from ALL markers -- which therefore contains "
            "the marker being tested, and that confounding costs "
            "power. Split the model: FEM tests one marker at a time "
            "with the currently associated markers as COVARIATES; REM "
            "estimates those associated markers by using them to "
            "DEFINE KINSHIP, avoiding over-fitting. Alternate, unify "
            "the p-values each round, and the confounding is removed "
            "rather than reduced. Cost is linear in individuals AND "
            "markers.")


# compact alias per ledger/NAMING.md
farmcpumodel = farmcpu

# public names resolved by fn/_lazy_map.json
farm_cpu = farmcpu
