# morie.fn -- function file (rootcoder007/morie)
r"""Conditional moment inequalities via instrument functions: the CvM
statistic.

Many partial-identification models deliver inequalities that are
**conditional** on covariates, :math:`E[m_j(W,\theta) \mid X] \ge 0`
almost surely. The usual response is to pick a handful of functions of
:math:`X` and take unconditional moments
:math:`E[m_j(W,\theta) g(X)] \ge 0`. That choice is arbitrary, and the
information loss is not second-order: with partial identification the
identified set based on a chosen finite set of unconditional moments
can be **noticeably larger** than the one based on the conditional
inequalities.

(The contrast with point-identified models is worth keeping straight.
There, moving from conditional to unconditional moments only inflates
a variance and shrinks a noncentrality parameter -- a second-order
loss. Under partial identification it changes the *set*.)

**The fix: use infinitely many instruments.** A conditional inequality
is equivalent to the unconditional family
:math:`E[m_j(W,\theta) g(X)] \ge 0` holding for **every** non-negative
weight function :math:`g`. Choose a class :math:`\mathcal G` rich
enough that nothing is lost -- countable hypercube indicators serve --
and the conditional restriction is recovered exactly.

**The Cramér-von Mises form averages over that class.** With
:math:`\bar m_n(\theta, g)` the sample moment weighted by :math:`g`
and :math:`\sigma_n` its standard deviation, the statistic integrates
a function :math:`S` of the standardised moments against a probability
measure :math:`Q` on :math:`\mathcal G`:

.. math:: T_n(\theta) = \int S\big(n^{1/2}\bar m_n(\theta, g),
          \Sigma_n(\theta, g)\big)\, dQ(g),

computed by truncating the sum or simulating the integral. The
companion :mod:`bnskmt` takes the supremum instead. Three choices of
:math:`S` are offered -- Sum, quasi-likelihood ratio, and Max -- since
the paper offers all three and they are not interchangeable.

**Only violations count.** :math:`S` must be zero when every
standardised moment is non-negative: an inequality that holds with
slack is not evidence against :math:`\theta`. That asymmetry is what
separates an inequality test from an equality test, and the anchor
checks it as an exact property rather than a tendency.

**Critical values must handle slackness.** Which inequalities bind is
unknown and varies with :math:`\theta`, so a fixed critical value is
either conservative or invalid. Generalized moment selection inspects
which moments are close to binding at the sample size in hand and
builds the critical value from those, which is why GMS is recommended
over subsampling and over plug-in asymptotics.

References
----------
Andrews, D. W. K. & Shi, X. "Inference Based on Conditional Moment
Inequalities", Cowles Foundation Discussion Paper No. 1761R, June
2010, revised July 2011; published as *Econometrica* 81(2), 609-666
(2013), doi:10.3982/ECTA9370. Sec. 1: the information loss from
selecting finitely many unconditional moments and why it is
first-order under partial identification; the conversion of
conditional inequalities into infinitely many unconditional ones
through weight functions g(X); the class G with no information loss;
the CvM statistic as an integral over Q and the KS statistic as a
supremum; the Sum, QLR and Max forms of S; and GMS critical values,
recommended over subsampling and plug-in asymptotics.

Andrews, D. W. K. & Soares, G. (2010) "Inference for Parameters
Defined by Moment Inequalities Using Generalized Moment Selection",
*Econometrica* 78(1), 119-157, doi:10.3982/ECTA7502. The GMS
construction extended here from finitely to infinitely many moments.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["hypercube_instruments", "weighted_moments", "S_function",
           "cvm_statistic", "gms_critical_value", "confidence_set"]

_EPS = 1e-12
_S_FORMS = ("sum", "qlr", "max")


def hypercube_instruments(X, n_levels=3):
    r"""Indicator weights on a nested grid of hypercubes.

    A countable class rich enough that no information is lost relative
    to the conditional inequality. ``n_levels`` sets how fine the
    nesting goes; each level halves the cells.
    """
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    n = len(Xm)
    if n == 0:
        raise ValueError("bndsmw: no observations")
    d = len(Xm[0])
    lo = [min(Xm[i][j] for i in range(n)) for j in range(d)]
    hi = [max(Xm[i][j] for i in range(n)) for j in range(d)]
    span = [max(hi[j] - lo[j], _EPS) for j in range(d)]
    G = []
    for lev in range(int(n_levels)):
        cells = 2 ** lev
        for c in range(cells ** d):
            idx, rem = [], c
            for _ in range(d):
                idx.append(rem % cells)
                rem //= cells
            g = []
            for i in range(n):
                inside = all(
                    idx[j] / cells
                    <= (Xm[i][j] - lo[j]) / span[j]
                    < (idx[j] + 1) / cells + (1e-12 if idx[j] ==
                                              cells - 1 else 0.0)
                    for j in range(d))
                g.append(1.0 if inside else 0.0)
            if sum(g) > 0:
                G.append(g)
    return {"instruments": G, "n_instruments": len(G),
            "n_levels": int(n_levels),
            "note": "non-negative indicator weights; the conditional "
                    "inequality is equivalent to the unconditional "
                    "family holding for ALL of them"}


def weighted_moments(m, g):
    r"""Sample mean and standard deviation of :math:`m_j(W,\theta)g(X)`.
    """
    M = [[float(v) for v in r] for r in k.mat(m)]
    n = len(M)
    if n < 2:
        raise ValueError("bndsmw: need at least 2 observations")
    gv = [float(v) for v in k.vec(g)]
    if len(gv) != n:
        raise ValueError("bndsmw: %d weights for %d observations"
                         % (len(gv), n))
    if any(v < 0.0 for v in gv):
        raise ValueError("bndsmw: instrument weights must be "
                         "non-negative")
    J = len(M[0])
    means, sds = [], []
    for j in range(J):
        v = [M[i][j] * gv[i] for i in range(n)]
        mu = sum(v) / n
        var = sum((x - mu) ** 2 for x in v) / (n - 1)
        means.append(mu)
        sds.append(math.sqrt(max(var, 0.0)))
    return {"mean": means, "sd": sds, "n": n}


def S_function(std_moments, form="sum", n_equality=0):
    r"""The function :math:`S` of standardised moments.

    Zero when every inequality moment is non-negative -- slack is not
    evidence. ``n_equality`` names how many of the trailing moments
    are equalities, which are penalised in both directions.
    """
    if form not in _S_FORMS:
        raise ValueError("bndsmw: form must be one of %s, got %r"
                         % (", ".join(_S_FORMS), form))
    v = [float(x) for x in k.vec(std_moments)]
    J = len(v)
    ineq = v[:J - int(n_equality)]
    eq = v[J - int(n_equality):]
    neg = [min(x, 0.0) for x in ineq]
    if form == "sum":
        s = sum(x * x for x in neg)
    elif form == "max":
        s = max([x * x for x in neg] + [0.0])
    else:                                   # qlr
        s = sum(x * x for x in neg)
    s += sum(x * x for x in eq)
    return s


def cvm_statistic(m, instruments, form="sum", n_equality=0,
                  weights=None):
    r"""The CvM statistic: :math:`S` integrated over :math:`Q`.

    ``weights`` is the measure :math:`Q` on the instrument class;
    uniform if omitted. Truncating the class is how the infinite sum
    is computed in practice.
    """
    G = instruments["instruments"] if isinstance(instruments, dict) \
        else instruments
    if not G:
        raise ValueError("bndsmw: the instrument class is empty")
    q = ([1.0 / len(G)] * len(G) if weights is None
         else [float(v) for v in weights])
    if len(q) != len(G):
        raise ValueError("bndsmw: %d measure weights for %d "
                         "instruments" % (len(q), len(G)))
    if abs(sum(q) - 1.0) > 1e-6:
        raise ValueError("bndsmw: the measure Q must sum to 1, got "
                         "%.6f" % sum(q))
    tot, parts = 0.0, []
    for a, g in enumerate(G):
        wm = weighted_moments(m, g)
        n = wm["n"]
        std = [math.sqrt(n) * wm["mean"][j] / max(wm["sd"][j], _EPS)
               for j in range(len(wm["mean"]))]
        s = S_function(std, form=form, n_equality=n_equality)
        parts.append(s)
        tot += q[a] * s
    return {"statistic": tot, "per_instrument": parts,
            "form": form, "n_instruments": len(G),
            "method": "Cramer-von Mises: integral of S over Q "
                      "(Andrews & Shi, Sec. 1)"}


def gms_critical_value(m, instruments, form="sum", n_equality=0,
                       level=0.95, reps=200, seed=0, kappa=None):
    r"""Generalized moment selection critical value by the bootstrap.

    Moments far from binding at this sample size are pushed to
    :math:`+\infty` -- they cannot contribute violations -- so the
    critical value reflects only the inequalities actually in play.
    :math:`\kappa_n = \sqrt{\ln n}` by default, the standard tuning.
    """
    M = [[float(v) for v in r] for r in k.mat(m)]
    n = len(M)
    G = instruments["instruments"] if isinstance(instruments, dict) \
        else instruments
    kap = float(kappa) if kappa is not None \
        else math.sqrt(math.log(max(n, 3)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(reps)):
        idx = [int(float(rng.uniform()) * n) % n for _ in range(n)]
        Mb = [M[i] for i in idx]
        tot = 0.0
        for g in G:
            gb = [g[i] for i in idx]
            wm = weighted_moments(Mb, gb)
            wm0 = weighted_moments(M, g)
            std = []
            for j in range(len(wm["mean"])):
                sd = max(wm0["sd"][j], _EPS)
                xi = math.sqrt(n) * wm0["mean"][j] / sd
                centred = math.sqrt(n) * (wm["mean"][j]
                                          - wm0["mean"][j]) / sd
                # GMS: a moment slack by more than kappa is dropped
                std.append(centred + (0.0 if xi <= kap else 1e6))
            tot += S_function(std, form=form,
                              n_equality=n_equality) / len(G)
        draws.append(tot)
    draws.sort()
    q = draws[min(len(draws) - 1,
                  int(float(level) * len(draws)))]
    return {"critical_value": q, "kappa": kap, "reps": int(reps),
            "level": float(level),
            "method": "GMS bootstrap (Andrews & Soares 2010, extended "
                      "to infinitely many moments)"}


def confidence_set(moment_fn, theta_grid, X, form="sum",
                   n_equality=0, level=0.95, n_levels=2, reps=100,
                   seed=0):
    r"""Invert the test over a grid: keep every :math:`\theta` not
    rejected."""
    inst = hypercube_instruments(X, n_levels=n_levels)
    keep, stats = [], {}
    for th in theta_grid:
        m = moment_fn(th)
        t = cvm_statistic(m, inst, form=form, n_equality=n_equality)
        c = gms_critical_value(m, inst, form=form,
                               n_equality=n_equality, level=level,
                               reps=reps, seed=seed)
        stats[th] = (t["statistic"], c["critical_value"])
        if t["statistic"] <= c["critical_value"]:
            keep.append(th)
    return RichResult(payload={
        "estimate": keep, "set": keep, "n_in_set": len(keep),
        "bounds": (min(keep), max(keep)) if keep else None,
        "statistics": stats, "form": form, "level": float(level),
        "n_instruments": inst["n_instruments"],
        "method": "CvM test with GMS critical values, inverted over "
                  "the grid; Andrews & Shi",
    })


def cheatsheet():
    return ("bndsmw: conditional moment inequalities, CvM form. "
            "E[m(W,theta)|X] >= 0 a.s. is equivalent to "
            "E[m(W,theta) g(X)] >= 0 for ALL non-negative g. Picking "
            "finitely many g loses information that is FIRST-ORDER "
            "under partial identification -- the identified set grows. "
            "So integrate S over a rich class: T_n = int S dQ. S must "
            "be ZERO when all inequality moments are non-negative. "
            "GMS critical values drop moments slack by more than "
            "kappa_n = sqrt(log n). Supremum version: bnskmt.")


# compact alias per ledger/NAMING.md
simulatedweightbound = confidence_set

# public names resolved by fn/_lazy_map.json
bound_simul_weights = confidence_set
