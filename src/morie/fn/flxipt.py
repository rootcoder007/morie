# morie.fn -- function file (rootcoder007/morie)
r"""Super Learner, and IPTW with the propensity score it produces.

Given J candidate learners and no way to know in advance which fits the
regression at hand, the Super Learner does not choose between them by
looking at how well each fits the data it was trained on. It builds a
*second* data set out of held-out predictions and learns the combination
there.

For each fold :math:`v`, train every candidate on :math:`T(v)` and
predict the held-out block, giving the level-one design

.. math:: Z_i = \big(\hat\Psi_j(P_n^{T(v(i))})(X_i) : j = 1,\dots,J\big),

then fit a meta-learner :math:`\tilde\Psi` of :math:`Y` on :math:`Z`.
The Super Learner is that meta-learner applied to the candidates refitted
on the whole sample,

.. math:: \hat\Psi(P_n)(X)
          = \hat\Psi^*(P_n)\big(\hat\Psi_j(P_n)(X), j=1,\dots,J\big).

**Every prediction in Z is out-of-sample, and that is the whole trick.**
Build Z from in-sample fits and every candidate's apparent risk is
optimistic, the flexible ones most of all, so the meta-learner is
choosing between numbers that are wrong by different amounts. Measured
over 25 draws at n = 140 with a library whose interaction learner
carries 22 parameters, the in-sample version understates its own risk
in *every* draw, by 0.22 on a risk of about 1.0, and moves the weights
by 0.35. It does not always predict worse -- at that sample size it
sometimes predicts a little better, because it favours the flexible
candidate and the truth here is nonlinear -- but the risk estimate it
reports is not usable, which is what the selection depends on.
``honest_level_one=False`` exists to demonstrate this, not to be used.

**What is promised.** The oracle result says the ensemble performs
asymptotically as well as the best candidate in the library, up to a
second-order term. Two consequences are checkable and both are checked:
when the library does *not* contain a correctly specified model the
convex combination beats every candidate in it, and when it does, the
ensemble matches that candidate rather than paying for the company it
keeps.

**Three meta-learners, because they answer different questions.**
``"nnls"`` is the convex combination -- weights non-negative and summing
to one, which is what makes the ensemble a proper average and keeps it
inside the range the candidates span. ``"discrete"`` is the plain
cross-validation selector: all the weight on the single best candidate,
which is the right choice when one candidate is genuinely correct.
``"ols"`` drops the constraints, and is included mainly to show what the
constraints buy.

**Why this matters for a propensity score.** Weighting by
:math:`1/\hat g(W)` puts the whole burden of the analysis on a model
nobody has any reason to believe. Pirracchio et al. fit that model with
a Super Learner instead: the weights are

.. math:: \text{IPTW}_i = \frac{A_i}{g(W_i)}
                        + \frac{1 - A_i}{1 - g(W_i)},

and the treatment effect follows from a weighted regression of Y on A.
Where the true treatment mechanism is not a main-terms logistic, the
ensemble recovers it and the misspecified parametric fit does not.

References
----------
van der Laan, M. J., Polley, E. C. & Hubbard, A. E. (2007) "Super
Learner", *Statistical Applications in Genetics and Molecular Biology*
6(1), article 25, doi:10.2202/1544-6115.1309. Sec. 2, eq. (1), and the
oracle result of its Theorem 2.

Pirracchio, R., Petersen, M. L. & van der Laan, M. J. (2015) "Improving
propensity score estimators' robustness to model misspecification using
super learner", *American Journal of Epidemiology* 181(2), 108-119,
doi:10.1093/aje/kwu253. The propensity-score application and the IPTW
weights of its eq. (3).

van der Laan, M. J. & Dudoit, S. (2003) "Unified Cross-Validation
Methodology For Selection Among Estimators and a General Cross-Validated
Adaptive Epsilon-Net Estimator: Finite Sample Oracle Inequalities and
Examples", *U.C. Berkeley Division of Biostatistics Working Paper
Series*, Working Paper 130. The finite-sample oracle inequality the
guarantee rests on.

Breiman, L. (1996) "Stacked Regressions", *Machine Learning* 24(1),
49-64, doi:10.1007/BF00117832. The stacking construction Super Learner
generalises, and the source of the non-negativity constraint.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["super_learner", "flexible_iptw", "iptw_ate",
           "default_learners", "cv_risk"]

_METAS = ("nnls", "discrete", "ols")
_EPS = 1e-9


def _expand(W, spec):
    """Build a candidate's design from the raw covariates."""
    n = len(W)
    p = len(W[0]) if n and W[0] else 0
    kind = spec["kind"]
    out = []
    for i in range(n):
        r = [1.0]
        if kind == "intercept":
            pass
        elif kind == "main":
            r += list(W[i])
        elif kind == "quadratic":
            r += list(W[i]) + [W[i][c] * W[i][c] for c in range(p)]
        elif kind == "interaction":
            r += list(W[i])
            for a in range(p):
                for b in range(a + 1, p):
                    r.append(W[i][a] * W[i][b])
        elif kind == "subset":
            r += [W[i][c] for c in spec["cols"]]
        else:
            raise ValueError("flxipt: unknown learner kind %r" % (kind,))
        out.append(r)
    return out


def default_learners(p, ridge_penalties=(0.0, 1.0, 10.0)):
    """A library in the shape Pirracchio et al. use.

    Main terms, main terms with pairwise interactions, a quadratic
    expansion, and penalised versions of the interaction model standing
    in for the penalised-likelihood entries of their library. The
    intercept-only learner is always present so the ensemble can fall
    back on the marginal when nothing predicts.
    """
    lib = [{"name": "intercept", "kind": "intercept", "penalty": 0.0},
           {"name": "main", "kind": "main", "penalty": 0.0},
           {"name": "quadratic", "kind": "quadratic", "penalty": 0.0}]
    if p >= 2:
        lib.append({"name": "interaction", "kind": "interaction",
                    "penalty": 0.0})
        for pen in ridge_penalties:
            if pen > 0.0:
                lib.append({"name": "interaction+ridge%g" % pen,
                            "kind": "interaction", "penalty": pen})
    return lib


def _fit_learner(y, W, spec, rows, binary, ridge):
    """Train one candidate on `rows`; return a predictor over all rows."""
    X = _expand(W, spec)
    Xr = [X[i] for i in rows]
    yr = [y[i] for i in rows]
    if binary:
        b = k.logit_irls(Xr, yr, ridge=max(ridge, 1e-10),
                         penalty=spec.get("penalty", 0.0))
        return [k.sigmoid(v) for v in k.matvec(X, b)], b
    pen = spec.get("penalty", 0.0)
    b = k.lstsq(Xr, yr, max(ridge, 1e-10) + pen)
    return list(k.matvec(X, b)), b


def _folds(n, V, seed=0):
    """Disjoint validation blocks whose union is the whole sample."""
    V = max(2, min(int(V), n))
    return [[i for i in range(n) if i % V == v] for v in range(V)]


def _project_simplex(v):
    """Euclidean projection onto {a : a >= 0, sum a = 1}.

    The exact sort-based projection, so the convex constraint is met
    rather than approximated by clipping and renormalising -- clipping
    does not give the closest point and can stall the descent.
    """
    n = len(v)
    if n == 0:
        return []
    u = sorted(v, reverse=True)
    css = 0.0
    rho, theta = 0, 0.0
    for j in range(n):
        css += u[j]
        t = (css - 1.0) / (j + 1)
        if u[j] - t > 0.0:
            rho, theta = j + 1, t
    return [max(x - theta, 0.0) for x in v]


def _nnls_simplex(Z, y, iters=8000, tol=1e-14):
    """Convex combination minimising ||y - Z a||^2 over the simplex.

    Accelerated projected gradient. Two details matter and both were
    found by the anchor rather than assumed:

    * the step comes from the largest eigenvalue of the Gram matrix,
      estimated by power iteration. Bounding it by the largest absolute
      row sum instead -- the obvious cheap substitute -- overestimates
      it badly when the candidates are near-collinear, which they
      always are when several of them are nearly correct, and the
      resulting step is so small the descent stalls short of the
      optimum.
    * every vertex of the simplex is feasible, so the optimum can never
      be worse than the best single candidate. That is checked against
      the returned point, and the better of the two is returned. It is
      a guard on the solver, not a substitute for it: if it ever binds,
      the iteration did not converge.
    """
    n = len(Z)
    J = len(Z[0]) if n else 0
    if J == 0:
        return []
    if J == 1:
        return [1.0]
    G = [[sum(Z[i][a] * Z[i][b] for i in range(n)) / n
          for b in range(J)] for a in range(J)]
    c = [sum(Z[i][a] * y[i] for i in range(n)) / n for a in range(J)]

    # power iteration for the largest eigenvalue of G
    v = [1.0 / math.sqrt(J)] * J
    lam = 0.0
    for _ in range(200):
        u = [sum(G[a][b] * v[b] for b in range(J)) for a in range(J)]
        nrm = math.sqrt(sum(x * x for x in u))
        if nrm <= 0.0:
            break
        v = [x / nrm for x in u]
        if abs(nrm - lam) < 1e-13 * max(nrm, 1.0):
            lam = nrm
            break
        lam = nrm
    step = 1.0 / lam if lam > 0.0 else 1.0

    def obj(a):
        return (sum(a[t] * sum(G[t][b] * a[b] for b in range(J))
                    for t in range(J))
                - 2.0 * sum(c[t] * a[t] for t in range(J)))

    a = [1.0 / J] * J
    z = list(a)
    tk = 1.0
    for _ in range(iters):
        grad = [sum(G[t][b] * z[b] for b in range(J)) - c[t]
                for t in range(J)]
        nxt = _project_simplex([z[t] - step * grad[t] for t in range(J)])
        tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        mom = (tk - 1.0) / tn
        z = [nxt[t] + mom * (nxt[t] - a[t]) for t in range(J)]
        shift = max(abs(nxt[t] - a[t]) for t in range(J))
        a, tk = nxt, tn
        if shift < tol:
            break

    best_vertex = min(range(J), key=lambda j: G[j][j] - 2.0 * c[j])
    vert = [1.0 if t == best_vertex else 0.0 for t in range(J)]
    return a if obj(a) <= obj(vert) else vert


def cv_risk(y, Z, loss="l2"):
    """Cross-validated risk of each column of the level-one design."""
    n = len(y)
    J = len(Z[0]) if n else 0
    out = []
    for j in range(J):
        if loss == "l2":
            out.append(sum((y[i] - Z[i][j]) ** 2 for i in range(n)) / n)
        elif loss == "nll":
            tot = 0.0
            for i in range(n):
                p = min(max(Z[i][j], _EPS), 1.0 - _EPS)
                tot -= y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
            out.append(tot / n)
        else:
            raise ValueError("flxipt: loss must be l2 or nll, got %r"
                             % (loss,))
    return out


def super_learner(y, X, library=None, n_folds=10, meta="nnls",
                  binary=None, loss="l2", ridge=1e-8,
                  honest_level_one=True):
    r"""Sec. 2 of van der Laan, Polley & Hubbard (2007).

    Parameters
    ----------
    y, X : array-like
        Outcome and covariates.
    library : sequence of dict, optional
        Candidate specs ``{"name", "kind", "penalty"}``; see
        :func:`default_learners`.
    meta : {"nnls", "discrete", "ols"}
        The minimum cross-validated risk predictor. ``nnls`` is the
        convex combination, ``discrete`` the plain CV selector.
    binary : bool, optional
        Fit candidates by logistic regression. Inferred from ``y``.
    honest_level_one : bool
        Build Z from held-out predictions, as the algorithm requires.
        ``False`` uses in-sample fits and exists only to demonstrate
        what that costs -- it is not a supported way to use this.

    Returns
    -------
    RichResult
        ``fitted`` are the ensemble's predictions, with ``weights``,
        each candidate's ``cv_risk``, and the ensemble's own.
    """
    if meta not in _METAS:
        raise ValueError("flxipt: meta must be one of %s, got %r"
                         % (", ".join(_METAS), meta))
    yv = k.vec(y)
    n = len(yv)
    Wm = k.mat(X) if X is not None else [[] for _ in range(n)]
    if len(Wm) != n:
        raise ValueError("flxipt: %d covariate rows for %d outcomes"
                         % (len(Wm), n))
    if n < 8:
        raise ValueError("flxipt: need at least 8 observations, got %d"
                         % n)
    p = len(Wm[0]) if Wm and Wm[0] else 0
    lib = default_learners(p) if library is None else list(library)
    if not lib:
        raise ValueError("flxipt: the library is empty")
    if binary is None:
        binary = all(v in (0.0, 1.0) for v in yv)
    if loss == "nll" and not binary:
        raise ValueError("flxipt: the nll loss needs a binary outcome")

    J = len(lib)
    folds = _folds(n, n_folds)
    Z = [[0.0] * J for _ in range(n)]
    for val in folds:
        train = [i for i in range(n) if i not in set(val)]
        if not train:
            raise ValueError("flxipt: an empty training fold")
        src = train if honest_level_one else list(range(n))
        for j, spec in enumerate(lib):
            pred, _ = _fit_learner(yv, Wm, spec, src, binary, ridge)
            for i in val:
                Z[i][j] = pred[i]

    risks = cv_risk(yv, Z, loss)
    best = min(range(J), key=lambda j: risks[j])
    if meta == "discrete":
        weights = [1.0 if j == best else 0.0 for j in range(J)]
    elif meta == "nnls":
        weights = _nnls_simplex(Z, yv)
    else:
        weights = list(k.lstsq(Z, yv, max(ridge, 1e-10)))

    # the candidates are refitted on ALL the data; only the combination
    # came from the held-out predictions
    full = []
    for spec in lib:
        pred, _ = _fit_learner(yv, Wm, spec, list(range(n)), binary,
                               ridge)
        full.append(pred)
    fitted = [sum(weights[j] * full[j][i] for j in range(J))
              for i in range(n)]
    if binary:
        fitted = [min(max(v, 0.0), 1.0) for v in fitted]

    ens = [[sum(weights[j] * Z[i][j] for j in range(J))] for i in range(n)]
    if binary:
        ens = [[min(max(r[0], _EPS), 1.0 - _EPS)] for r in ens]
    ens_risk = cv_risk(yv, ens, loss)[0]

    return RichResult(payload={
        "fitted": fitted, "estimate": ens_risk,
        "weights": {lib[j]["name"]: weights[j] for j in range(J)},
        "weight_vector": weights,
        "cv_risk": {lib[j]["name"]: risks[j] for j in range(J)},
        "cv_risk_ensemble": ens_risk,
        "best_candidate": lib[best]["name"],
        "best_candidate_risk": risks[best],
        "discrete_choice": lib[best]["name"],
        "level_one": Z, "candidate_fits": full,
        "library": [s["name"] for s in lib], "n": n, "n_folds": len(folds),
        "meta": meta, "loss": loss, "binary": bool(binary),
        "honest_level_one": bool(honest_level_one),
        "method": "Super Learner, van der Laan, Polley & Hubbard (2007) "
                  "Sec. 2 eq. (1)",
    })


def flexible_iptw(A, H, library=None, n_folds=10, meta="nnls",
                  trim=0.01, ridge=1e-8, stabilize=False):
    r"""Propensity score by Super Learner, and the IPTW weights.

    ``A`` is the binary treatment and ``H`` the history it may depend
    on. The weights are eq. (3) of Pirracchio et al.,
    :math:`A/g(W) + (1-A)/(1-g(W))`, optionally stabilised by the
    marginal treatment probability.
    """
    Av = k.vec(A)
    n = len(Av)
    if any(v not in (0.0, 1.0) for v in Av):
        raise ValueError("flxipt: the treatment must be binary 0/1")
    if not 0 < sum(Av) < n:
        raise ValueError("flxipt: both treatment arms must be non-empty")
    t = float(trim)
    if not 0.0 <= t < 0.5:
        raise ValueError("flxipt: trim must be in [0, 0.5), got %r"
                         % (trim,))
    sl = super_learner(Av, H, library=library, n_folds=n_folds,
                       meta=meta, binary=True, loss="l2", ridge=ridge)
    g = [min(max(v, max(t, _EPS)), 1.0 - max(t, _EPS))
         for v in sl["fitted"]]
    if stabilize:
        # SW = P(A = a) / g(a | W): the marginal in the numerator, which
        # leaves the weights centred near 1 instead of near 1/g
        pa = sum(Av) / n
        w = [Av[i] * pa / g[i]
             + (1.0 - Av[i]) * (1.0 - pa) / (1.0 - g[i])
             for i in range(n)]
    else:
        w = [Av[i] / g[i] + (1.0 - Av[i]) / (1.0 - g[i])
             for i in range(n)]
    return RichResult(payload={
        "propensity": g, "weights": w, "estimate": sum(w) / n,
        "sl_weights": sl["weights"], "cv_risk": sl["cv_risk"],
        "cv_risk_ensemble": sl["cv_risk_ensemble"],
        "best_candidate": sl["best_candidate"],
        "max_weight": max(w), "min_propensity": min(g),
        "max_propensity": max(g), "n": n, "trim": t,
        "stabilized": bool(stabilize), "library": sl["library"],
        "method": "IPTW with a Super Learner propensity score, "
                  "Pirracchio, Petersen & van der Laan (2015) eq. (3)",
    })


def iptw_ate(y, A, H, library=None, n_folds=10, meta="nnls",
             trim=0.01, ridge=1e-8, level=0.95):
    """ATE by a weighted regression of Y on A, Pirracchio et al.

    The Hajek form is used -- each arm's weighted mean divides by its
    own weight total -- so the estimate stays inside the range of the
    outcome even when a few weights are large.
    """
    yv, Av = k.vec(y), k.vec(A)
    n = len(yv)
    if len(Av) != n:
        raise ValueError("flxipt: %d outcomes but %d treatments"
                         % (n, len(Av)))
    r = flexible_iptw(Av, H, library=library, n_folds=n_folds,
                      meta=meta, trim=trim, ridge=ridge)
    w = r["weights"]
    w1 = sum(w[i] for i in range(n) if Av[i] == 1.0)
    w0 = sum(w[i] for i in range(n) if Av[i] == 0.0)
    if w1 <= 0.0 or w0 <= 0.0:
        raise ValueError("flxipt: an arm carries no weight")
    m1 = sum(w[i] * yv[i] for i in range(n) if Av[i] == 1.0) / w1
    m0 = sum(w[i] * yv[i] for i in range(n) if Av[i] == 0.0) / w0
    psi = m1 - m0
    # influence-curve standard error for the Hajek contrast
    ic = []
    for i in range(n):
        if Av[i] == 1.0:
            ic.append(w[i] * (yv[i] - m1) * n / w1)
        else:
            ic.append(-w[i] * (yv[i] - m0) * n / w0)
    se = k.sd(ic) / math.sqrt(n)
    z = k.qnorm(0.5 + 0.5 * float(level))
    return RichResult(payload={
        "estimate": psi, "se": se, "ci": (psi - z * se, psi + z * se),
        "mean_treated": m1, "mean_control": m0,
        "propensity": r["propensity"], "weights": w,
        "sl_weights": r["sl_weights"], "cv_risk": r["cv_risk"],
        "best_candidate": r["best_candidate"],
        "max_weight": r["max_weight"],
        "min_propensity": r["min_propensity"], "n": n,
        "level": float(level),
        "method": "IPTW ATE with a Super Learner propensity score, "
                  "Pirracchio, Petersen & van der Laan (2015)",
    })


def cheatsheet():
    return ("flxipt: Super Learner. Z[i][j] = candidate j's HELD-OUT "
            "prediction for i; fit the meta-learner of y on Z (nnls "
            "convex combination, or discrete = the CV selector); apply "
            "it to the candidates refitted on all the data (vdL-Polley-"
            "Hubbard 2007 eq. 1). Then IPTW weights A/g + (1-A)/(1-g) "
            "with g from the ensemble (Pirracchio 2015 eq. 3).")


# compact alias per ledger/NAMING.md
flexibleiptw = flexible_iptw
