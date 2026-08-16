"""Sequential (on-line) estimation of a density under a DP mixture filter.

Batch MCMC for a Dirichlet-process mixture sees the whole sample before
it says anything. A filter sees y_1, then y_2, and must have an answer
after each one, never revisiting what it already processed. That
constraint is not a weakness to be apologised for -- it is the whole
point when the data arrives as a stream and a decision is due before the
next observation.

The state is the allocation of the points seen so far. The component
parameters are NOT part of the state: with a conjugate
normal / inverse-gamma prior they can be integrated out exactly, so a
cluster is carried as three sufficient statistics (count, sum, sum of
squares) and its predictive density for the next point is a Student t.
That is Rao-Blackwellisation, and it is what makes a particle filter over
allocations feasible at all -- particles over (allocation, parameters)
would need a Metropolis move per step and would degenerate.

At step t, a particle holding an allocation of y_1..y_{t-1} extends it by
assigning y_t. The Polya urn gives the prior:

    P(join cluster j) proportional to n_j
    P(start a new one) proportional to alpha

and the conjugate predictive gives the likelihood. Two proposals, both
implemented:

  "optimal"   sample the new allocation from the EXACT conditional --
              prior times likelihood, normalised -- and multiply the
              weight by that normalising constant. This is the minimum
              variance proposal for this model (MacEachern, Clyde and
              Liu; Fearnhead). Every particle survives every step with a
              well-behaved weight.
  "prior"     sample from the Polya urn alone and weight by the
              likelihood. Cheaper per step and much worse: a particle
              that guesses the cluster badly gets a tiny weight, so the
              population degenerates and the filter spends its life
              resampling. Included because the difference between the
              two is the single most instructive thing about this
              algorithm, and it should be measurable rather than
              asserted -- the harness anchors on the effective sample
              size of each.

Resampling happens when the effective sample size falls below a
threshold. Three schemes, all exact:

  "multinomial"  n independent draws. Simplest, highest variance.
  "stratified"   one draw per stratum of width 1/n.
  "systematic"   ONE draw, then a regular comb. Lowest variance of the
                 three and the usual default; the correlation it
                 introduces between selections is exactly what removes
                 the variance.

References
  MacEachern, S.N., Clyde, M. and Liu, J.S. (1999) "Sequential
    importance sampling for nonparametric Bayes models: the next
    generation." Canadian Journal of Statistics 27(2), 251-267.
  Fearnhead, P. (2004) "Particle filters for mixture models with an
    unknown number of components." Statistics and Computing 14(1),
    11-21.
  Caron, F., Doucet, A. and Gottardo, R. (2012) "On-line changepoint
    detection and parameter estimation with application to genomic
    data." Statistics and Computing 22(2), 579-595. The ledger entry
    for this module cites this work as 2017; the paper is 2012, and the
    year is corrected here rather than repeated.
  Escobar, M.D. and West, M. (1995) JASA 90(430), 577-588 -- the
    conjugate normal / inverse-gamma DP mixture the predictive comes
    from.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["sn_pseudo_estimate", "snpest", "student_t_logpdf",
           "cluster_predictive", "PROPOSALS", "RESAMPLERS", "cheatsheet"]

PROPOSALS = ("optimal", "prior")
RESAMPLERS = ("systematic", "stratified", "multinomial")


def student_t_logpdf(x, df, loc, scale2):
    """log density of a Student t with df degrees of freedom.

    Written out rather than taken from a distribution library: the two
    languages carry separate implementations of the t density and they
    do not agree in the last digits.
    """
    z = (x - loc) * (x - loc) / (df * scale2)
    return (_w.lgamma(0.5 * (df + 1.0)) - _w.lgamma(0.5 * df)
            - 0.5 * math.log(df * math.pi * scale2)
            - 0.5 * (df + 1.0) * math.log(1.0 + z))


def cluster_predictive(x, n, s, ss, m0, kappa0, a0, b0):
    """log p(x | cluster with sufficient statistics (n, sum, sumsq)).

    n = 0 gives the prior predictive, which is what a brand-new cluster
    uses -- the same expression, so there is no separate code path to
    get wrong.
    """
    if n > 0:
        ybar = s / n
        sse = ss - s * s / n
        kn = kappa0 + n
        mn = (kappa0 * m0 + s) / kn
        an = a0 + 0.5 * n
        bn = (b0 + 0.5 * sse
              + 0.5 * kappa0 * n * (ybar - m0) * (ybar - m0) / kn)
    else:
        kn, mn, an, bn = kappa0, m0, a0, b0
    df = 2.0 * an
    scale2 = bn * (kn + 1.0) / (an * kn)
    return student_t_logpdf(x, df, mn, scale2)


def _resample(rng, weights, scheme):
    """Indices of the resampled particles, weights assumed normalised."""
    n = len(weights)
    if scheme == "multinomial":
        us = sorted(float(rng.uniform()) for _ in range(n))
    elif scheme == "stratified":
        us = [(k + float(rng.uniform())) / n for k in range(n)]
    elif scheme == "systematic":
        u0 = float(rng.uniform())
        us = [(k + u0) / n for k in range(n)]
    else:
        raise ValueError("resampler must be one of %r" % (RESAMPLERS,))
    out = []
    acc = 0.0
    j = 0
    for k in range(n):
        acc += weights[k]
        while j < n and us[j] <= acc:
            out.append(k)
            j += 1
    while len(out) < n:
        out.append(n - 1)
    return out


def _ess(weights):
    """Effective sample size, 1 / sum w^2 for normalised weights."""
    return 1.0 / _w.csum(v * v for v in weights)


def sn_pseudo_estimate(y_stream, alpha=1.0, n_particles=100,
                       proposal="optimal", resampler="systematic",
                       ess_threshold=0.5, m0=None, kappa0=0.01, a0=2.0,
                       b0=None, seed=1, grid=None, seed_stats=True):
    """On-line DP-mixture filter over a stream of observations.

    Parameters
    ----------
    y_stream : sequence
        The observations, in arrival order. The order MATTERS: this is a
        filter, not a batch method, and the answer after t points is a
        function of the first t.
    alpha : float
        Dirichlet-process concentration.
    n_particles : int
        Particles carried.
    proposal : str
        "optimal" or "prior".
    resampler : str
        "systematic", "stratified" or "multinomial".
    ess_threshold : float
        Resample when the effective sample size falls below this
        fraction of the particle count.
    m0, kappa0, a0, b0 : float
        Normal / inverse-gamma prior. m0 and b0 default to the stream's
        own mean and variance; passing them explicitly is what a genuine
        on-line run would do, since a filter is not supposed to have
        seen the future.
    seed : int
        Seed for the generator shared with the R arm.
    grid : sequence or None
        Points at which the filtered predictive density is recorded
        after the last observation.
    seed_stats : bool
        Take m0 and b0 from the whole stream when they are not given.
        Set False to use 0 and 1 instead and keep the run strictly
        causal.

    Returns
    -------
    RichResult
        The running log marginal likelihood, the expected number of
        clusters after each point, the ESS trace, the resampling times,
        and the filtered predictive density on the grid.

    References
    ----------
    MacEachern, Clyde and Liu (1999) Canad. J. Statist. 27(2), 251-267;
    Fearnhead (2004) Statist. Comput. 14(1), 11-21;
    Caron, Doucet and Gottardo (2012) Statist. Comput. 22(2), 579-595.
    """
    if proposal not in PROPOSALS:
        raise ValueError("proposal must be one of %r" % (PROPOSALS,))
    if resampler not in RESAMPLERS:
        raise ValueError("resampler must be one of %r" % (RESAMPLERS,))
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    ys = [float(v) for v in y_stream]
    T = len(ys)
    if T < 2:
        raise ValueError("need at least two observations")
    P = int(n_particles)
    if P < 1:
        raise ValueError("need at least one particle")
    if seed_stats:
        mean = _w.csum(ys) / T
        var = _w.csum((v - mean) * (v - mean) for v in ys) / (T - 1)
    else:
        mean, var = 0.0, 1.0
    if m0 is None:
        m0 = mean
    if b0 is None:
        b0 = var * (a0 - 1.0) if a0 > 1.0 else var

    rng = _core._SplitMix64(seed)

    # Each particle is a list of clusters, each a [count, sum, sumsq].
    parts = [[] for _ in range(P)]
    logw = [0.0] * P
    loglik_trace = []
    ess_trace = []
    clusters_trace = []
    resampled_at = []

    for t in range(T):
        x = ys[t]
        incr = [0.0] * P
        for p in range(P):
            cl = parts[p]
            K = len(cl)
            lp = []
            for j in range(K):
                lp.append(math.log(cl[j][0])
                          + cluster_predictive(x, cl[j][0], cl[j][1],
                                               cl[j][2], m0, kappa0, a0,
                                               b0))
            lp.append(math.log(alpha)
                      + cluster_predictive(x, 0, 0.0, 0.0, m0, kappa0,
                                           a0, b0))
            norm = _w.logsumexp(lp)
            if proposal == "optimal":
                # Draw from the exact conditional; the weight increment
                # is the normalising constant, which is p(y_t | past).
                probs = [math.exp(v - norm) for v in lp]
                u = float(rng.uniform())
                acc = 0.0
                pick = K
                for j in range(K + 1):
                    acc += probs[j]
                    if u <= acc:
                        pick = j
                        break
                incr[p] = norm - math.log(alpha + t)
            else:
                # Draw from the Polya urn alone; the weight increment is
                # the likelihood of the cluster that was drawn.
                tot = alpha + t
                u = float(rng.uniform()) * tot
                acc = 0.0
                pick = K
                for j in range(K):
                    acc += cl[j][0]
                    if u <= acc:
                        pick = j
                        break
                incr[p] = (lp[pick] - math.log(cl[pick][0] if pick < K
                                               else alpha))
            if pick == K:
                cl.append([1.0, x, x * x])
            else:
                cl[pick][0] += 1.0
                cl[pick][1] += x
                cl[pick][2] += x * x

        # Update and normalise the weights in log space.
        for p in range(P):
            logw[p] += incr[p]
        lnorm = _w.logsumexp(logw)
        wts = [math.exp(v - lnorm) for v in logw]
        # lnorm IS the running log marginal likelihood: the weights
        # start at zero and a resample rewrites them as lnorm - log(P)
        # each, which leaves their total untouched. Recording it here,
        # before the resample, is what keeps it the right quantity.
        loglik_trace.append(lnorm)
        e = _ess(wts)
        ess_trace.append(e)
        clusters_trace.append(_w.csum(wts[p] * len(parts[p])
                                      for p in range(P)))

        if e < ess_threshold * P:
            idx = _resample(rng, wts, resampler)
            parts = [[list(c) for c in parts[i]] for i in idx]
            logw = [lnorm - math.log(P)] * P
            resampled_at.append(t)

    lnorm = _w.logsumexp(logw)
    wts = [math.exp(v - lnorm) for v in logw]

    if grid is None:
        lo = min(ys) - math.sqrt(var)
        hi = max(ys) + math.sqrt(var)
        grid = [lo + (hi - lo) * k / 20.0 for k in range(21)]
    grid = [float(g) for g in grid]

    dens = []
    for g in grid:
        acc = []
        for p in range(P):
            cl = parts[p]
            lp = [math.log(c[0]) + cluster_predictive(g, c[0], c[1], c[2],
                                                      m0, kappa0, a0, b0)
                  for c in cl]
            lp.append(math.log(alpha)
                      + cluster_predictive(g, 0, 0.0, 0.0, m0, kappa0,
                                           a0, b0))
            acc.append(wts[p] * math.exp(_w.logsumexp(lp)
                                         - math.log(alpha + T)))
        dens.append(_w.csum(acc))

    return RichResult(payload={
        "grid": grid,
        "density": dens,
        "log_marginal": loglik_trace[-1],
        "log_marginal_trace": loglik_trace,
        "ess": ess_trace,
        "final_ess": ess_trace[-1],
        "mean_ess": _w.csum(ess_trace) / T,
        "clusters": clusters_trace,
        "final_clusters": clusters_trace[-1],
        "resampled_at": resampled_at,
        "n_resamples": len(resampled_at),
        "T": T,
        "n_particles": P,
        "proposal": proposal,
        "resampler": resampler,
        "alpha": float(alpha),
        "prior": {"m0": m0, "kappa0": kappa0, "a0": a0, "b0": b0},
        "seed": int(seed),
        "estimate": clusters_trace[-1],
        "method": "sequential DP-mixture particle filter",
    })


snpest = sn_pseudo_estimate


def cheatsheet():
    return ("snpest: on-line DP-mixture particle filter. proposals "
            + ", ".join(PROPOSALS) + "; resamplers "
            + ", ".join(RESAMPLERS))
