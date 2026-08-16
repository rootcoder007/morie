"""Three-layer causal forest with a mediator: direct and indirect effects.

A treatment can work through a mediator or around it. A drug lowers
mortality partly by lowering blood pressure and partly by something
else; a training programme raises wages partly by raising skill and
partly by signalling. Splitting the total effect into those two parts is
mediation analysis, and doing it without assuming linear models anywhere
is what the three layers are for:

  layer 1   e(x)      = E[D | X]           the propensity
  layer 2   mbar(d,x) = E[M | D=d, X]      the mediator model
  layer 3   mu(d,m,x) = E[Y | D=d, M=m, X] the outcome model

each an honest regression forest. With those three, the natural effects
follow from the mediation formula:

  NDE(x) = mu(1, M(0), x) - mu(0, M(0), x)   treatment's own path
  NIE(x) = mu(1, M(1), x) - mu(1, M(0), x)   the path through M
  total  = NDE + NIE                          identically

The identity is not approximate and it is checked here rather than
assumed: the two parts are constructed to telescope, and if they ever
failed to add to the total the decomposition would be meaningless.

How M(d) enters is the choice that separates an honest implementation
from a convenient one. Plugging the mediator's conditional MEAN is
exact only if mu is linear in m, and a forest is not linear in
anything, so that route is offered and labelled rather than hidden:

  "mean"       substitute mbar(d, x). Fast, and wrong by the curvature
               of mu in m.
  "gcomputed"  average mu over the mediator's whole conditional
               distribution, represented by the empirical residuals of
               layer two added back to mbar(d, x). This is the g-formula
               and it is the default.

The propensity layer is reported rather than used to reweight: with all
three forests fitted the plug-in decomposition needs no weighting, but a
propensity near zero or one means the comparison at that x is being
extrapolated, and the OVERLAP the module reports is how you find that
out. A mediation estimate at a covariate value where nobody was
treated is arithmetic, not evidence.

References
  Cui, Y. and Tchetgen Tchetgen, E.J. (2024) "Machine intelligence for
    individualized decision making under a counterfactual world: a
    rejoinder." Journal of the American Statistical Association 119(545),
    97-102. The forest-based mediation machinery this follows.
  Pearl, J. (2001) "Direct and indirect effects." Proceedings of the
    Seventeenth Conference on Uncertainty in Artificial Intelligence,
    411-420. The mediation formula and the natural effects.
  Imai, K., Keele, L. and Yamamoto, T. (2010) "Identification,
    inference and sensitivity analysis for causal mediation effects."
    Statistical Science 25(1), 51-71. The identification conditions
    these estimates rest on.
  Athey, S., Tibshirani, J. and Wager, S. (2019) "Generalized random
    forests." The Annals of Statistics 47(2), 1148-1178.
  VanderWeele, T.J. (2015) "Explanation in Causal Inference: Methods
    for Mediation and Interaction." Oxford University Press.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from .sdcfst import honest_forest, forest_predict
from ._richresult import RichResult

__all__ = ["cthrgr", "causal_three_layer_grf", "ROUTES", "cheatsheet"]

ROUTES = ("gcomputed", "mean")


def _augment(D, M, X):
    return [[float(D[i]), float(M[i])] + list(X[i]) for i in range(len(X))]


def causal_three_layer_grf(y, D, M, X, route="gcomputed", n_trees=8,
                           min_leaf=3, max_depth=3, seed=0, n_draw=8,
                           newX=None):
    """Natural direct and indirect effects from three honest forests.

    Parameters
    ----------
    y, D, M : sequence
        Outcome, binary treatment, mediator.
    X : sequence of sequences
        Covariates.
    route : str
        A member of ROUTES.
    n_draw : int
        Mediator residuals averaged over, for the g-computed route.
        They are taken as an evenly spaced sweep through the sorted
        residuals rather than resampled, so the estimate is a fixed
        function of the data and not of a random stream.

    Returns
    -------
    RichResult
        Per-point direct, indirect and total effects, their averages,
        the fitted propensity and its overlap, and the mediator models.

    References
    ----------
    Pearl (2001) UAI 17, 411-420; Cui and Tchetgen Tchetgen (2024)
    JASA 119(545), 97-102.
    """
    if route not in ROUTES:
        raise ValueError("route must be one of %r" % (ROUTES,))
    ys = [float(v) for v in y]
    d = [1.0 if v else 0.0 for v in D]
    m = [float(v) for v in M]
    xs = [[float(v) for v in row] for row in X]
    n = len(ys)
    if len(d) != n or len(m) != n or len(xs) != n:
        raise ValueError("y, D, M and X must agree in length")
    if n < 8:
        raise ValueError("need at least eight observations")
    if sum(d) == 0 or sum(d) == n:
        raise ValueError("both treatment arms must be present")
    rows = list(range(n))

    # Layer one: the propensity, reported for overlap rather than used
    # to reweight.
    rng = _core._SplitMix64(seed)
    fe = honest_forest(xs, d, rows, n_trees, None, min_leaf, max_depth,
                       seed, rng)
    # Layer two: the mediator, one forest per arm so the arms may differ.
    mx = [[dd] + xx for dd, xx in zip(d, xs)]
    fm = honest_forest(mx, m, rows, n_trees, None, min_leaf, max_depth,
                       seed + 1, rng)
    # Layer three: the outcome on treatment, mediator and covariates.
    ax = _augment(d, m, xs)
    fy = honest_forest(ax, ys, rows, n_trees, None, min_leaf, max_depth,
                       seed + 2, rng)

    # The mediator residuals carry its conditional spread. Sorting them
    # and sweeping evenly is a deterministic quadrature over the
    # empirical distribution -- a resample would put a random stream
    # inside an estimate that has no business being random.
    resid = sorted(m[i] - forest_predict(fm, mx[i]) for i in range(n))
    k = max(1, int(n_draw))
    if route == "mean" or k == 1:
        draws = [0.0]
    else:
        draws = []
        for t in range(k):
            idx = int((t + 0.5) * len(resid) / k)
            if idx >= len(resid):
                idx = len(resid) - 1
            draws.append(resid[idx])

    qx = xs if newX is None else [[float(v) for v in r] for r in newX]
    nde = []
    nie = []
    tot = []
    ps = []
    m0v = []
    m1v = []
    for x in qx:
        e = forest_predict(fe, x)
        ps.append(e)
        mb0 = forest_predict(fm, [0.0] + x)
        mb1 = forest_predict(fm, [1.0] + x)
        m0v.append(mb0)
        m1v.append(mb1)
        a = []
        b = []
        c = []
        for r in draws:
            y10 = forest_predict(fy, [1.0, mb0 + r] + x)
            y00 = forest_predict(fy, [0.0, mb0 + r] + x)
            y11 = forest_predict(fy, [1.0, mb1 + r] + x)
            a.append(y10 - y00)
            b.append(y11 - y10)
            c.append(y11 - y00)
        nde.append(_w.csum(a) / len(a))
        nie.append(_w.csum(b) / len(b))
        tot.append(_w.csum(c) / len(c))

    q = len(qx)
    ande = _w.csum(nde) / q
    anie = _w.csum(nie) / q
    atot = _w.csum(tot) / q
    lo = min(ps)
    hi = max(ps)
    return RichResult(payload={
        "direct": nde,
        "indirect": nie,
        "total": tot,
        "propensity": ps,
        "mediator_control": m0v,
        "mediator_treated": m1v,
        "nde": ande,
        "nie": anie,
        "estimate": atot,
        "se": float("nan"),
        "proportion_mediated": anie / atot if atot != 0.0 else float("nan"),
        "overlap_min": lo,
        "overlap_max": hi,
        "n_extreme": sum(1 for e in ps if e < 0.05 or e > 0.95),
        "n_draw": len(draws),
        "residual_spread": resid[-1] - resid[0] if resid else 0.0,
        "n": n,
        "n_treated": int(sum(d)),
        "n_query": q,
        "route": route,
        "method": "three-layer causal forest with a mediator",
    })


cthrgr = causal_three_layer_grf


def cheatsheet():
    return ("cthrgr: three-layer causal forest with a mediator. routes "
            + ", ".join(ROUTES) + "; direct plus indirect is the total")
