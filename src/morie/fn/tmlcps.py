# morie.fn -- function file (rootcoder007/morie)
r"""Doubly robust estimation of a continuous-treatment effect curve.

Kennedy, Ma, McHugh & Small (2017) estimate the effect curve
:math:`\theta(a) = E(Y^a)` for a continuous treatment in two stages.
Their Theorem 1 gives the doubly robust mapping

.. math:: \xi(Z; \pi, \mu) =
          \frac{Y - \mu(L, A)}{\pi(A \mid L)}
          \int_L \pi(A \mid l)\, dP(l)
          + \int_L \mu(l, A)\, dP(l),

whose defining property is that
:math:`E\{\xi(Z; \pi, \mu) \mid A = a\} = \theta(a)` **if either**
:math:`\pi` **or** :math:`\mu` **is correct**. So the recipe is:

1. fit the nuisance functions -- the conditional treatment density
   :math:`\pi(a \mid l)` and the outcome regression
   :math:`\mu(l, a)`;
2. build the pseudo-outcome and regress it on the treatment by any
   nonparametric method.

The paper analyses a kernel version of step 2 with cross-validated
bandwidth, so that is the default here; a local-linear variant and a
plain polynomial fit are also available, since the paper is explicit
that "a wide variety of flexible methods could be used in our Step 2".

**Why the two integrals are over the sample and not over the point.**
:math:`\int_L \pi(A \mid l)\,dP(l)` is the *marginal* treatment density
evaluated at this observation's own treatment value, and
:math:`\int_L \mu(l, A)\,dP(l)` is the outcome regression standardized
over the covariate distribution at that same treatment value. Both are
averages over everyone else's covariates with this row's treatment held
fixed. Using the row's own covariates instead -- the natural misreading
-- collapses the first ratio to 1 and destroys the double robustness,
which the anchor detects.

This is not TMLE despite the module name: there is no fluctuation step
and no targeting. It is the doubly robust pseudo-outcome regression of
the cited paper, and calling it what it is seemed better than making
the name true by implementing something else.

References
----------
Kennedy, E. H., Ma, Z., McHugh, M. D. & Small, D. S. (2017)
"Non-parametric methods for doubly robust estimation of continuous
treatment effects", *Journal of the Royal Statistical Society, Series
B* 79(4), 1229-1245, doi:10.1111/rssb.12212; arXiv:1507.00747.
Theorem 1 and Sec. 3.2.

Hirano, K. & Imbens, G. W. (2004) "The propensity score with continuous
treatments", in Gelman, A. & Meng, X.-L. (eds.), *Applied Bayesian
Modeling and Causal Inference from Incomplete-Data Perspectives*,
Wiley, 73-84, doi:10.1002/0470090456.ch7 -- the generalized propensity
score that :math:`\pi(a \mid l)` is.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tmle_continuous_treatment", "pseudo_outcome",
           "effect_curve"]

_FITS = ("kernel", "locallinear", "polynomial")


def pseudo_outcome(y, A, X, ridge=1e-8):
    r"""Kennedy et al. Theorem 1's xi(Z; pi, mu), one value per row.

    Returns ``(xi, info)``. The nuisances are a Gaussian conditional
    treatment density and a linear outcome regression with a treatment
    interaction; both are the working models, and the point of the
    construction is that only one of them has to be right.
    """
    yv, av = k.vec(y), k.vec(A)
    n = len(yv)
    if len(av) != n:
        raise ValueError("pseudo_outcome: %d outcomes but %d treatments"
                         % (n, len(av)))
    Xm = k.mat(X) if X is not None else [[] for _ in range(n)]
    if len(Xm) != n:
        raise ValueError("pseudo_outcome: %d outcomes but %d covariate "
                         "rows" % (n, len(Xm)))
    p = len(Xm[0]) if Xm and Xm[0] else 0

    # pi(a | l): Gaussian, mean linear in l
    Zt = k.design(Xm if p else None, n)
    bt = k.lstsq(Zt, av, ridge)
    mu_a = k.matvec(Zt, bt)
    res = [av[i] - mu_a[i] for i in range(n)]
    s2 = sum(r * r for r in res) / max(1, n - len(bt))
    if s2 <= 0.0:
        raise ValueError("pseudo_outcome: the treatment model fits the "
                         "dose exactly, so pi(A|L) is degenerate")
    c = 1.0 / math.sqrt(2.0 * math.pi * s2)

    def pi_at(a, j):
        """pi(a | L_j), the density at dose `a` for row j's covariates."""
        r = a - mu_a[j]
        return c * math.exp(-0.5 * r * r / s2)

    # mu(l, a): linear with a treatment-covariate interaction
    Zy = [[av[i]] + list(Xm[i]) + [av[i] * Xm[i][q] for q in range(p)]
          for i in range(n)]
    by = k.lstsq(k.design(Zy, n), yv, ridge)

    def mu_at(a, j):
        row = [1.0, a] + list(Xm[j]) + [a * Xm[j][q] for q in range(p)]
        return sum(by[t] * row[t] for t in range(len(by)))

    xi, marg, stand = [], [], []
    for i in range(n):
        a_i = av[i]
        # integral over L of pi(a_i | l) dP(l): the MARGINAL density at
        # this row's treatment, averaged across everyone's covariates
        m = sum(pi_at(a_i, j) for j in range(n)) / n
        # integral over L of mu(l, a_i) dP(l): the outcome regression
        # standardized over the covariate distribution at that treatment
        s = sum(mu_at(a_i, j) for j in range(n)) / n
        den = pi_at(a_i, i)
        if den <= 0.0:
            raise ValueError(
                "pseudo_outcome: pi(A|L) is zero at observation %d, so "
                "positivity fails and xi is undefined" % i)
        xi.append((yv[i] - mu_at(a_i, i)) * m / den + s)
        marg.append(m)
        stand.append(s)
    return xi, {"marginal_density": marg, "standardized_mu": stand,
                "treatment_coef": bt, "treatment_sigma2": s2,
                "outcome_coef": by, "pi_obs": [pi_at(av[i], i)
                                               for i in range(n)]}


def effect_curve(xi, A, grid, fit="kernel", bandwidth=None,
                 n_folds=5):
    """Stage 2: regress the pseudo-outcome on the treatment.

    "kernel" is the Nadaraya-Watson estimator the paper analyses,
    "locallinear" the local-linear version that is less biased at the
    boundary, and "polynomial" a global cubic for when the curve is
    genuinely smooth and the sample is small. Bandwidth by
    cross-validation when not given, as in Sec. 3.3.
    """
    if fit not in _FITS:
        raise ValueError("effect_curve: fit must be one of %r, got %r"
                         % (_FITS, fit))
    xv, av = list(xi), k.vec(A)
    n = len(xv)
    gr = [float(v) for v in k.vec(grid)]
    if fit == "polynomial":
        X = [[av[i], av[i] ** 2, av[i] ** 3] for i in range(n)]
        f = k.wls(X, xv, [1.0] * n)
        b = f["coef"]
        return [b[0] + b[1] * g + b[2] * g * g + b[3] * g ** 3
                for g in gr], {"coef": b, "bandwidth": None}
    if bandwidth is None:
        bandwidth = _cv_bandwidth(xv, av, fit, n_folds)
    h = float(bandwidth)
    if h <= 0.0:
        raise ValueError("effect_curve: bandwidth must be positive, got "
                         "%r" % (bandwidth,))
    out = [_smooth_at(xv, av, g, h, fit) for g in gr]
    return out, {"bandwidth": h, "coef": None}


def _kern(u):
    return math.exp(-0.5 * u * u)


def _smooth_at(xv, av, g, h, fit):
    n = len(xv)
    w = [_kern((av[i] - g) / h) for i in range(n)]
    sw = sum(w)
    if sw <= 0.0:
        return float("nan")
    if fit == "kernel":
        return sum(w[i] * xv[i] for i in range(n)) / sw
    # local linear: weighted least squares of xi on (a - g)
    s1 = sum(w[i] * (av[i] - g) for i in range(n))
    s2 = sum(w[i] * (av[i] - g) ** 2 for i in range(n))
    t0 = sum(w[i] * xv[i] for i in range(n))
    t1 = sum(w[i] * (av[i] - g) * xv[i] for i in range(n))
    det = sw * s2 - s1 * s1
    if abs(det) < 1e-300:
        return t0 / sw
    return (s2 * t0 - s1 * t1) / det


def _cv_bandwidth(xv, av, fit, n_folds):
    """K-fold cross-validation over a log-spaced grid, Sec. 3.3."""
    n = len(xv)
    spread = max(av) - min(av)
    if spread <= 0.0:
        raise ValueError("_cv_bandwidth: the treatment is constant")
    grid = [spread * f for f in
            (0.02, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8)]
    folds = [[i for i in range(n) if i % n_folds == f]
             for f in range(int(n_folds))]
    best, best_h = None, grid[0]
    for h in grid:
        err = 0.0
        for f in folds:
            tr = [i for i in range(n) if i not in set(f)]
            xtr = [xv[i] for i in tr]
            atr = [av[i] for i in tr]
            for i in f:
                pred = _smooth_at(xtr, atr, av[i], h, fit)
                if pred != pred:
                    err += 1e12
                else:
                    err += (xv[i] - pred) ** 2
        if best is None or err < best:
            best, best_h = err, h
    return best_h


def tmle_continuous_treatment(y, A, X, a_grid=None, fit="kernel",
                              bandwidth=None, n_folds=5):
    r"""The effect curve theta(a) = E(Y^a) for a continuous treatment.

    Parameters
    ----------
    y, A, X : array-like
        Outcome, continuous treatment, covariates.
    a_grid : array-like, optional
        Where to evaluate the curve. Defaults to 21 points spanning the
        observed treatment range.
    fit : {"kernel", "locallinear", "polynomial"}
        Stage-2 regression.

    Returns
    -------
    RichResult
        ``estimate`` is the average derivative of the curve -- the
        per-unit effect -- with ``curve`` on ``grid``.

    Examples
    --------
    A confounded dose whose true effect curve has slope 1.5::

        r = tmle_continuous_treatment(y, dose, L)
        r["estimate"], r["curve"]
    """
    av = k.vec(A)
    if len(set(av)) < 3:
        raise ValueError(
            "tmle_continuous_treatment: the treatment takes %d distinct "
            "values; this estimates a continuous effect curve and a "
            "binary exposure belongs elsewhere" % len(set(av)))
    xi, info = pseudo_outcome(y, A, X)
    if a_grid is None:
        lo, hi = min(av), max(av)
        a_grid = [lo + (hi - lo) * t / 20.0 for t in range(21)]
    curve, cinfo = effect_curve(xi, av, a_grid, fit=fit,
                                bandwidth=bandwidth, n_folds=n_folds)
    gr = [float(v) for v in k.vec(a_grid)]
    slopes = [(curve[t + 1] - curve[t]) / (gr[t + 1] - gr[t])
              for t in range(len(gr) - 1) if gr[t + 1] != gr[t]]
    est = sum(slopes) / len(slopes) if slopes else float("nan")
    n = len(av)
    xbar = sum(xi) / n
    se = math.sqrt(sum((v - xbar) ** 2 for v in xi) / (n * (n - 1))) \
        if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": est,
        "se": se,
        "curve": curve, "grid": gr, "slopes": slopes,
        "pseudo_outcome": xi,
        "bandwidth": cinfo["bandwidth"],
        "marginal_density": info["marginal_density"],
        "standardized_mu": info["standardized_mu"],
        "pi_obs": info["pi_obs"],
        "fit": fit, "n": n,
        "method": "doubly robust effect curve for a continuous "
                  "treatment, Kennedy, Ma, McHugh & Small (2017) "
                  "Theorem 1 and Sec. 3.2",
    })


def cheatsheet():
    return ("tmlcps: continuous-treatment effect curve theta(a)=E(Y^a) "
            "(Kennedy-Ma-McHugh-Small 2017). Stage 1 pseudo-outcome "
            "xi = (Y-mu(L,A))/pi(A|L) * int pi(A|l)dP(l) + int "
            "mu(l,A)dP(l); stage 2 kernel / local-linear / polynomial "
            "regression of xi on A. Doubly robust in (pi, mu).")


# compact alias per ledger/NAMING.md
tmlecontinuoustreatment = tmle_continuous_treatment
