"""Slice samplers for Dirichlet-process mixtures (Walker; Kalli-Griffin-Walker).

A Dirichlet-process mixture has infinitely many components. Every
practical sampler has to make that finite somehow, and the older answer
-- truncate the stick after K terms and hope K was big enough -- changes
the model to make the computation possible. The slice sampler does the
opposite: it adds an auxiliary variable that makes the number of
components needed at each sweep FINITE AND RANDOM, while leaving the
posterior exactly the one the infinite model implies. Nothing is
truncated; the truncation point is resampled.

The trick. Write the mixture as sum_k w_k f(y | theta_k) with the
stick-breaking weights w_k = v_k prod_{l<k} (1 - v_l), v_k ~ Beta(1,
alpha). Introduce u_i and define the joint

    p(y_i, u_i, d_i) = 1{u_i < w_{d_i}} f(y_i | theta_{d_i})

Marginalising u_i over (0, w_{d_i}) gives back w_{d_i} f(y_i |
theta_{d_i}), so the model is unchanged. But CONDITIONAL on u_i only
components with w_k > u_i can be chosen, and since the weights sum to
one only finitely many qualify. That is Walker's sampler.

Kalli, Griffin and Walker's refinement replaces the random bound w_{d_i}
with a DETERMINISTIC decreasing sequence xi_k = (1 - kappa) kappa^(k-1):

    p(y_i, u_i, d_i) = 1{u_i < xi_{d_i}} (w_{d_i} / xi_{d_i})
                         f(y_i | theta_{d_i})

Now the number of components needed is a deterministic function of
min(u), so it cannot be driven to something huge by one unlucky small
weight -- which is the failure mode of the original when a component's
weight is tiny but its slice variable happens to land under it. The
price is the extra ratio w/xi in the label weights, and kappa is a
tuning knob: small kappa cuts the component count and costs mixing.

Both are implemented, both are exact, and `route` chooses. They target
the same posterior, so on the same data with the same prior they must
agree in distribution -- which is what the parity harness checks by
comparing posterior summaries rather than paths.

The component model is the conjugate normal / inverse-gamma:

    theta_k = (mu_k, s2_k),   s2_k ~ InvGamma(a0, b0),
    mu_k | s2_k ~ N(m0, s2_k / kappa0)

so every parameter draw is closed form and no Metropolis step is needed
anywhere. alpha may be held fixed or given Escobar and West's gamma
prior and updated by their two-component mixture.

References
  Walker, S.G. (2007) "Sampling the Dirichlet mixture model with
    slices." Communications in Statistics -- Simulation and
    Computation 36(1), 45-54.
  Kalli, M., Griffin, J.E. and Walker, S.G. (2011) "Slice sampling
    mixture models." Statistics and Computing 21(1), 93-105.
  Escobar, M.D. and West, M. (1995) "Bayesian density estimation and
    inference using mixtures." Journal of the American Statistical
    Association 90(430), 577-588. Section 6: the alpha update.
  Ishwaran, H. and James, L.F. (2001) "Gibbs sampling methods for
    stick-breaking priors." JASA 96(453), 161-173. The blocked
    conditionals the parameter updates follow.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["slice_break_dp", "slicebreakdp", "slbpdg", "stick_weights",
           "mixture_density", "ROUTES", "cheatsheet"]

ROUTES = ("walker", "kalli_griffin_walker")


def stick_weights(v):
    """Stick-breaking weights from the beta variates.

    w_k = v_k prod_{l<k} (1 - v_l). Accumulated as a running product
    rather than recomputed, which is both the cheap way and the way the
    R arm can match term for term.
    """
    w = []
    rest = 1.0
    for vk in v:
        w.append(vk * rest)
        rest = rest * (1.0 - vk)
    return w, rest


def _dnorm(x, mu, s2):
    d = x - mu
    return math.exp(-0.5 * d * d / s2) / math.sqrt(2.0 * math.pi * s2)


def mixture_density(x, w, mu, s2):
    """sum_k w_k N(x; mu_k, s2_k) over the components carried."""
    return _w.csum(w[k] * _dnorm(x, mu[k], s2[k]) for k in range(len(w)))


def _draw_theta(rng, ys, m0, kappa0, a0, b0):
    """Conjugate normal / inverse-gamma draw for one component.

    With no members this is a draw from the prior, which is what the
    slice sampler needs for the components it has to invent to cover the
    slice -- they are not fitted to anything, so their parameters must
    come from the prior and not from some neighbour.
    """
    n = len(ys)
    if n:
        ybar = _w.csum(ys) / n
        ss = _w.csum((v - ybar) * (v - ybar) for v in ys)
        kn = kappa0 + n
        mn = (kappa0 * m0 + n * ybar) / kn
        an = a0 + 0.5 * n
        bn = b0 + 0.5 * ss + 0.5 * kappa0 * n * (ybar - m0) * (ybar - m0) / kn
    else:
        kn, mn, an, bn = kappa0, m0, a0, b0
    # s2 first, then mu: the mean's variance depends on the draw of s2,
    # so the order is forced, and it fixes the stream position too.
    s2 = bn / float(rng.gamma(an, 1.0))
    mu = mn + math.sqrt(s2 / kn) * float(rng.normal())
    return mu, s2


def _categorical(rng, weights):
    """Inverse-CDF draw on UNNORMALISED weights, one uniform each.

    The cumulative sum is a plain running sum in both arms rather than
    a compensated one, because the comparison is against a uniform draw
    scaled by the same total -- consistency between the two sums is what
    matters, not their accuracy.
    """
    tot = _w.csum(weights)
    if tot <= 0.0:
        return -1
    u = float(rng.uniform()) * tot
    acc = 0.0
    for k in range(len(weights)):
        acc += weights[k]
        if u <= acc:
            return k
    return len(weights) - 1


def _xi(kappa, k):
    """The deterministic bound xi_k = (1 - kappa) kappa^(k-1).

    Built by repeated multiplication, not kappa ** (k-1): R's `^` on an
    integer exponent is repeated squaring and Python's `**` calls
    pow(), and the two part company in the last bit exactly where the
    comparison xi_k > u_i decides how many components to carry.
    """
    p = 1.0
    for _ in range(k):
        p *= kappa
    return (1.0 - kappa) * p


def slice_break_dp(y, alpha=1.0, n_iter=500, burn=None, thin=1,
                   route="walker", kappa=0.5, m0=None, kappa0=0.01,
                   a0=2.0, b0=None, seed=1, max_components=200,
                   grid=None, alpha_update=None, alpha_a=2.0,
                   alpha_b=1.0, keep_draws=False):
    """Slice-sampled Dirichlet-process mixture of normals.

    Parameters
    ----------
    y : sequence
        The data.
    alpha : float
        Dirichlet-process concentration. The starting value when it is
        being updated.
    n_iter : int
        Total sweeps, including burn-in.
    burn : int or None
        Sweeps discarded. Half of n_iter by default.
    thin : int
        Keep every `thin`-th sweep after burn-in.
    route : str
        "walker" for the random bound u_i < w_{d_i}, or
        "kalli_griffin_walker" for the deterministic xi_k.
    kappa : float
        The geometric rate of xi_k, used only by the second route.
        Smaller carries fewer components and mixes worse.
    m0, kappa0, a0, b0 : float
        Normal / inverse-gamma prior. m0 defaults to the sample mean and
        b0 to the sample variance, which puts the prior on the scale of
        the data instead of on the scale of whatever units it arrived
        in.
    seed : int
        Seed for the generator shared with the R arm.
    max_components : int
        A hard ceiling on the components carried in one sweep. Reaching
        it is reported in `hit_ceiling` rather than silently truncating
        the model into a different one.
    grid : sequence or None
        Points at which the posterior mean density is accumulated.
    alpha_update : str or None
        None to hold alpha fixed, or "escobar_west" for the gamma-prior
        update.
    alpha_a, alpha_b : float
        Shape and rate of that gamma prior.
    keep_draws : bool
        Also return the component state of every retained sweep --
        weights, means and variances. Downstream modules that need a
        posterior of some functional of the mixture (a quantile, a
        tail probability) work from these rather than re-running a
        sampler of their own.

    Returns
    -------
    RichResult
        Posterior summaries: the mean density on the grid, the
        distribution of the number of occupied clusters, the retained
        alpha draws, and per-sweep diagnostics.

    References
    ----------
    Walker (2007) Comm. Statist. Simul. Comput. 36(1), 45-54;
    Kalli, Griffin and Walker (2011) Statist. Comput. 21(1), 93-105;
    Escobar and West (1995) JASA 90(430), 577-588.
    """
    if route not in ROUTES:
        raise ValueError("route must be one of %r" % (ROUTES,))
    if not (0.0 < kappa < 1.0):
        raise ValueError("kappa must lie strictly inside (0, 1)")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    ys = [float(v) for v in y]
    n = len(ys)
    if n < 2:
        raise ValueError("need at least two observations")
    if burn is None:
        burn = n_iter // 2
    ybar = _w.csum(ys) / n
    var = _w.csum((v - ybar) * (v - ybar) for v in ys) / (n - 1)
    if m0 is None:
        m0 = ybar
    if b0 is None:
        b0 = var * (a0 - 1.0) if a0 > 1.0 else var
    if grid is None:
        lo = min(ys) - math.sqrt(var)
        hi = max(ys) + math.sqrt(var)
        grid = [lo + (hi - lo) * k / 20.0 for k in range(21)]
    grid = [float(g) for g in grid]

    rng = _core._SplitMix64(seed)

    # One component to start, everybody in it.
    v = [float(rng.beta(1.0, alpha))]
    mu, s2 = _draw_theta(rng, ys, m0, kappa0, a0, b0)
    mus = [mu]
    s2s = [s2]
    d = [0] * n

    dens = [0.0] * len(grid)
    kept = 0
    n_clusters = []
    n_components = []
    alphas = []
    hit_ceiling = False
    max_carried = 0
    draws = []

    for it in range(int(n_iter)):
        w, rest = stick_weights(v)

        # 1. the slice variables
        if route == "walker":
            u = [float(rng.uniform()) * w[d[i]] for i in range(n)]
        else:
            u = [float(rng.uniform()) * _xi(kappa, d[i]) for i in range(n)]
        umin = min(u)

        # 2. grow the stick until no component beyond it can be chosen.
        #    Walker: until the leftover mass is below the smallest slice.
        #    KGW: until xi_K itself is, which does not depend on the
        #    weights at all and is the point of the variant.
        while True:
            if route == "walker":
                if rest <= umin or len(v) >= max_components:
                    break
            else:
                if _xi(kappa, len(v)) <= umin or len(v) >= max_components:
                    break
            vk = float(rng.beta(1.0, alpha))
            v.append(vk)
            mk, sk = _draw_theta(rng, [], m0, kappa0, a0, b0)
            mus.append(mk)
            s2s.append(sk)
            w, rest = stick_weights(v)
        if len(v) >= max_components:
            hit_ceiling = True
        K = len(v)
        if K > max_carried:
            max_carried = K

        # 3. labels, from the components the slice admits
        for i in range(n):
            wt = [0.0] * K
            for k in range(K):
                if route == "walker":
                    if w[k] > u[i]:
                        wt[k] = _dnorm(ys[i], mus[k], s2s[k])
                else:
                    xk = _xi(kappa, k)
                    if xk > u[i]:
                        wt[k] = (w[k] / xk) * _dnorm(ys[i], mus[k], s2s[k])
            j = _categorical(rng, wt)
            if j >= 0:
                d[i] = j

        # 4. the sticks, from the counts above and beyond each index
        counts = [0] * K
        for i in range(n):
            counts[d[i]] += 1
        after = [0] * K
        run = 0
        for k in range(K - 1, -1, -1):
            after[k] = run
            run += counts[k]
        for k in range(K):
            v[k] = float(rng.beta(1.0 + counts[k], alpha + after[k]))

        # 5. the component parameters
        members = [[] for _ in range(K)]
        for i in range(n):
            members[d[i]].append(ys[i])
        for k in range(K):
            mus[k], s2s[k] = _draw_theta(rng, members[k], m0, kappa0, a0,
                                         b0)

        occupied = sum(1 for c in counts if c > 0)

        # 6. alpha, by Escobar and West's two-component mixture
        if alpha_update == "escobar_west":
            eta = float(rng.beta(alpha + 1.0, float(n)))
            odds = (alpha_a + occupied - 1.0) / (n * (alpha_b - math.log(eta)))
            pi_eta = odds / (1.0 + odds)
            if float(rng.uniform()) < pi_eta:
                alpha = float(rng.gamma(alpha_a + occupied,
                                        1.0 / (alpha_b - math.log(eta))))
            else:
                alpha = float(rng.gamma(alpha_a + occupied - 1.0,
                                        1.0 / (alpha_b - math.log(eta))))
        elif alpha_update is not None:
            raise ValueError('alpha_update must be None or "escobar_west"')

        if it >= burn and (it - burn) % thin == 0:
            kept += 1
            n_clusters.append(occupied)
            n_components.append(K)
            alphas.append(alpha)
            w, rest = stick_weights(v)
            for gi in range(len(grid)):
                dens[gi] += mixture_density(grid[gi], w, mus, s2s)
            if keep_draws:
                draws.append({"w": list(w), "mu": list(mus),
                              "s2": list(s2s), "rest": rest})

    if kept == 0:
        raise ValueError("burn-in consumed every sweep")
    dens = [t / kept for t in dens]
    mean_clusters = _w.csum(float(c) for c in n_clusters) / kept
    mean_alpha = _w.csum(alphas) / kept
    tab = {}
    for c in n_clusters:
        tab[c] = tab.get(c, 0) + 1
    modal = max(sorted(tab), key=lambda c: (tab[c], -c))

    payload = {
        "grid": grid,
        "density": dens,
        "density_integral": _w.simpson(
            lambda x: _interp(grid, dens, x), grid[0], grid[-1], 200),
        "n_clusters": n_clusters,
        "mean_clusters": mean_clusters,
        "modal_clusters": modal,
        "cluster_table": [[c, tab[c]] for c in sorted(tab)],
        "n_components": n_components,
        "max_components_carried": max_carried,
        "hit_ceiling": hit_ceiling,
        "alpha_draws": alphas,
        "mean_alpha": mean_alpha,
        "kept": kept,
        "n": n,
        "route": route,
        "kappa": float(kappa),
        "prior": {"m0": m0, "kappa0": kappa0, "a0": a0, "b0": b0},
        "seed": int(seed),
        "estimate": mean_clusters,
        "method": "slice-sampled Dirichlet-process mixture",
    }
    if keep_draws:
        payload["draws"] = draws
    return RichResult(payload=payload)


def _interp(xs, ys, x):
    """Linear interpolation, used only to integrate the density grid."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


slicebreakdp = slice_break_dp
slbpdg = slice_break_dp


def cheatsheet():
    return ("slbpdg: slice-sampled Dirichlet-process mixture. routes "
            + ", ".join(ROUTES))
