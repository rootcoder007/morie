# morie.fn -- function file (rootcoder007/morie)
r"""Sequential Monte Carlo: the bootstrap particle filter.

A state-space model has a hidden state that evolves and an observation
that depends on it. Outside the linear-Gaussian case the filtering
density has no closed form, so it is carried by a swarm of weighted
particles instead.

Algorithm 1 of the pomp paper, one observation at a time:

1. propagate each particle through the state process,
   :math:`X^P_{n,j} \sim f_{X_n|X_{n-1}}(\cdot \mid X^F_{n-1,j})`;
2. weight it by how well it explains the observation,
   :math:`w(n,j) = f_{Y_n|X_n}(y^*_n \mid X^P_{n,j})`;
3. resample.

**The likelihood comes out for free, and it is unbiased.** The average
weight at each step estimates the one-step predictive density, so the
product over :math:`n` estimates the full likelihood -- and it is
unbiased for the likelihood itself, not for its logarithm. That
distinction is not pedantry: :math:`\log` of an unbiased estimate is
biased *downward* by Jensen, so a filter with too few particles reports
a likelihood that is systematically too low, and comparing two models
run at different particle counts compares the counts as much as the
models. The anchor measures that downward bias rather than mentioning
it.

**Systematic resampling, not multinomial.** Draw one uniform and take
:math:`J` evenly spaced points through the cumulative weights. The
count each particle receives then differs from :math:`J w_j` by less
than one, *always* -- multinomial resampling only achieves that in
expectation, and the extra variance it adds is pure loss. The anchor
checks the deterministic bound holds for every particle.

**Particle depletion is the failure mode.** When one weight dominates,
the swarm collapses to a single distinct particle and every subsequent
estimate is that one trajectory wearing an average. The effective
sample size :math:`(\sum w)^2 / \sum w^2` is reported at every step
because a filter that has degenerated still returns numbers.

References
----------
King, A. A., Nguyen, D. & Ionides, E. L. (2016) "Statistical Inference
for Partially Observed Markov Processes: The R Package pomp", *Journal
of Statistical Software* 69(12), 1-43, doi:10.18637/jss.v069.i12.
Algorithm 1 (SMC) and Algorithm 2 (systematic resampling).

Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993) "Novel approach
to nonlinear/non-Gaussian Bayesian state estimation", *IEE Proceedings
F (Radar and Signal Processing)* 140(2), 107-113,
doi:10.1049/ip-f-2.1993.0015. The bootstrap filter itself. NOT
retrievable at the time of writing -- the algorithm here follows pomp's
Algorithm 1, which states it in full.

Doucet, A. & Johansen, A. M. (2011) "A Tutorial on Particle Filtering
and Smoothing: Fifteen Years Later", in Crisan, D. & Rozovskii, B.
(eds.) *The Oxford Handbook of Nonlinear Filtering*, Oxford University
Press, 656-704.

Kalman, R. E. (1960) "A New Approach to Linear Filtering and Prediction
Problems", *Journal of Basic Engineering* 82(1), 35-45,
doi:10.1115/1.3662552. The closed form the filter must reproduce on a
linear-Gaussian model, which is how it is checked here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["systematic_resample", "effective_sample_size",
           "particle_filter", "kalman_filter_1d"]

_EPS = 1e-300


def effective_sample_size(weights):
    r""":math:`(\sum w)^2 / \sum w^2` -- how many particles are really
    contributing."""
    s1 = sum(weights)
    s2 = sum(v * v for v in weights)
    if s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2


def systematic_resample(weights, u=None, rng=None):
    r"""Algorithm 2: one uniform, J evenly spaced points.

    The count particle j receives differs from :math:`J w_j` by less
    than 1 -- deterministically, not in expectation, which is the whole
    reason to prefer it over multinomial resampling.
    """
    J = len(weights)
    tot = sum(weights)
    if tot <= 0.0:
        raise ValueError("prtcl: all particle weights are zero; the "
                         "filter has lost the signal")
    w = [v / tot for v in weights]
    if u is None:
        u = float(rng.uniform()) if rng is not None else 0.5
    if not 0.0 <= u < 1.0:
        raise ValueError("prtcl: the offset must lie in [0, 1), got %r"
                         % (u,))
    idx = []
    cum = w[0]
    j = 0
    for m in range(J):
        pos = (m + u) / J
        while pos > cum and j < J - 1:
            j += 1
            cum += w[j]
        idx.append(j)
    return idx


def particle_filter(y, n_particles, init, step, loglik, seed=0,
                    resample_threshold=1.0, systematic=True):
    r"""Algorithm 1, returning the filtered mean and the likelihood.

    Parameters
    ----------
    init : callable
        ``init(rng) -> state`` for one particle.
    step : callable
        ``step(state, t, rng) -> state``, the process simulator.
    loglik : callable
        ``loglik(state, obs, t) -> float``, the measurement density.
    resample_threshold : float
        Resample when ESS falls below this fraction of J. The default
        of 1.0 resamples every step, as Algorithm 1 does.

    Returns
    -------
    RichResult
        ``loglik`` is the log of the UNBIASED likelihood estimate --
        biased downward as a log, which is reported alongside the
        effective sample sizes so the degeneracy is visible.
    """
    obs = list(y)
    N = len(obs)
    J = int(n_particles)
    if J < 2:
        raise ValueError("prtcl: need at least 2 particles, got %d" % J)
    if N == 0:
        raise ValueError("prtcl: no observations")
    if not 0.0 < resample_threshold <= 1.0:
        raise ValueError("prtcl: resample_threshold must be in (0, 1], "
                         "got %r" % (resample_threshold,))
    rng = np.random.default_rng(seed)
    parts = [init(rng) for _ in range(J)]
    ll = 0.0
    means, esss, resampled = [], [], []
    for n in range(N):
        parts = [step(parts[j], n, rng) for j in range(J)]
        lw = [loglik(parts[j], obs[n], n) for j in range(J)]
        mx = max(lw)
        if mx == float("-inf"):
            raise ValueError("prtcl: every particle has zero likelihood "
                             "at observation %d" % n)
        w = [math.exp(v - mx) for v in lw]
        tot = sum(w)
        # log mean weight: the one-step predictive density
        ll += mx + math.log(tot / J)
        ess = effective_sample_size(w)
        esss.append(ess)
        means.append(sum(w[j] * _scalar(parts[j])
                         for j in range(J)) / tot)
        if ess < resample_threshold * J:
            idx = (systematic_resample(w, rng=rng) if systematic
                   else _multinomial(w, rng))
            parts = [parts[i] for i in idx]
            resampled.append(True)
        else:
            resampled.append(False)
    return RichResult(payload={
        "estimate": means, "filtered_mean": means, "loglik": ll,
        "ess": esss, "min_ess": min(esss), "resampled": resampled,
        "n_particles": J, "n_obs": N, "systematic": bool(systematic),
        "particles": parts,
        "method": "bootstrap particle filter, King, Nguyen & Ionides "
                  "(2016) Algorithm 1 with systematic resampling "
                  "(Algorithm 2)",
    })


def _scalar(state):
    return float(state[0]) if hasattr(state, "__len__") else float(state)


def _multinomial(w, rng):
    tot = sum(w)
    cum, acc = [], 0.0
    for v in w:
        acc += v / tot
        cum.append(acc)
    out = []
    for _ in range(len(w)):
        u = float(rng.uniform())
        j = 0
        while j < len(cum) - 1 and u > cum[j]:
            j += 1
        out.append(j)
    return out


def kalman_filter_1d(y, a, q, c, r, m0=0.0, p0=1.0):
    r"""The closed form for the scalar linear-Gaussian model.

    :math:`x_n = a x_{n-1} + N(0,q)`, :math:`y_n = c x_n + N(0,r)`.
    Provided so the particle filter can be checked against an exact
    answer rather than against itself.
    """
    m, p = float(m0), float(p0)
    means, ll = [], 0.0
    for obs in y:
        m = a * m
        p = a * a * p + q
        s = c * c * p + r
        v = obs - c * m
        ll += -0.5 * (math.log(2.0 * math.pi * s) + v * v / s)
        gain = p * c / s
        m = m + gain * v
        p = (1.0 - gain * c) * p
        means.append(m)
    return means, ll


def cheatsheet():
    return ("prtcl: propagate, weight by the measurement density, "
            "resample (pomp Alg. 1). Mean weight per step gives an "
            "UNBIASED likelihood -- so its LOG is biased DOWNWARD by "
            "Jensen, and comparing models at different particle counts "
            "compares the counts. Systematic resampling (Alg. 2) gives "
            "each particle a count within 1 of J*w_j deterministically. "
            "Watch ESS: a degenerate filter still returns numbers.")


# compact alias per ledger/NAMING.md
particlefilter = particle_filter
