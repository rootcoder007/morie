# morie.fn -- function file (rootcoder007/morie)
r"""Likelihood-based Metropolis-Hastings for a compartmental epidemic.

**The model.** A closed SIR population of size :math:`N`, integrated
in discrete steps:

.. math:: \Delta S_t = -\beta S_t I_t / N, \qquad
          \Delta I_t = \beta S_t I_t / N - \gamma I_t,

so new infections in step :math:`t` are
:math:`\lambda_t = \beta S_t I_t \Delta t / N`. Counts are observed
with Poisson error, giving

.. math:: \ell(\beta, \gamma) = \sum_t
          \bigl(y_t \log \lambda_t - \lambda_t - \log y_t!\bigr).

**Why MCMC and not maximisation.** :math:`\beta` and :math:`\gamma`
are badly identified from incidence alone -- what the early curve
pins down is the ratio :math:`R_0 = \beta/\gamma`, not either
separately, so the likelihood has a long curved ridge. A point
estimate on that ridge is close to meaningless without the posterior
around it, which is O'Neill and Roberts' argument for sampling rather
than optimising.

**The sampler.** Random-walk Metropolis on
:math:`(\log\beta, \log\gamma)` -- log scale so the positivity
constraint is automatic and the proposal is symmetric, which leaves
the Hastings ratio equal to the posterior ratio. Priors are
log-normal.

**Parity note.** The chain is driven by the SplitMix64 stream shared
by the Python and R sides, and every random draw is taken from it in
the same order. The two implementations therefore produce the
*identical* chain, draw for draw, not merely chains that agree in
distribution. That is what makes this module checkable rather than
merely plausible.

**What is reported.** The chain, the acceptance rate, posterior means
and quantiles, and :math:`R_0` per draw. A random-walk sampler whose
acceptance rate has collapsed toward zero or one is not exploring;
the rate is returned rather than assumed adequate.

References
----------
O'Neill, P. D. & Roberts, G. O. (1999) "Bayesian inference for
partially observed stochastic epidemics", *Journal of the Royal
Statistical Society: Series A* 162(1), 121-129,
doi:10.1111/1467-985X.00125. Bayesian estimation of epidemic
transmission parameters by MCMC rather than by maximisation, and the
treatment of the transmission and removal rates as jointly uncertain.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sir_incidence", "poisson_loglik", "likelihood_mcmc_epi"]


def sir_incidence(beta, gamma, S0, I0, N, n_steps, dt=1.0):
    r"""Expected new infections per step from the deterministic SIR."""
    b, g = float(beta), float(gamma)
    if b <= 0 or g <= 0:
        raise ValueError("likemc: beta and gamma must be positive")
    if float(N) <= 0:
        raise ValueError("likemc: the population size must be positive")
    S, I = float(S0), float(I0)
    out = []
    for _ in range(int(n_steps)):
        lam = b * S * I / float(N) * float(dt)
        lam = max(lam, 1e-12)
        rem = g * I * float(dt)
        out.append(lam)
        S = max(S - lam, 0.0)
        I = max(I + lam - rem, 0.0)
    return out


def poisson_loglik(observed, expected):
    r"""Poisson log-likelihood, dropping no terms."""
    y = [float(v) for v in observed]
    lam = [float(v) for v in expected]
    if len(y) != len(lam):
        raise ValueError("likemc: %d observations but %d expected "
                         "counts" % (len(y), len(lam)))
    if any(v < 0 for v in y):
        raise ValueError("likemc: a count cannot be negative")
    tot = 0.0
    for i in range(len(y)):
        L = max(lam[i], 1e-12)
        tot += y[i] * math.log(L) - L - math.lgamma(y[i] + 1.0)
    return tot


def _lognorm_lpdf(x, mu, sigma):
    if x <= 0:
        return float("-inf")
    z = (math.log(x) - mu) / sigma
    return -math.log(x * sigma * math.sqrt(2.0 * math.pi)) - 0.5 * z * z


def likelihood_mcmc_epi(model, data, priors, n_iter, seed=1,
                        step=0.15, burn=0):
    r"""Random-walk Metropolis on (log beta, log gamma).

    ``model`` carries ``S0``, ``I0``, ``N`` and optionally ``dt``;
    ``priors`` carries ``beta_mu``, ``beta_sigma``, ``gamma_mu``,
    ``gamma_sigma`` on the log scale.
    """
    y = [float(v) for v in data]
    if len(y) < 2:
        raise ValueError("likemc: need at least two observed counts")
    S0 = float(model["S0"])
    I0 = float(model["I0"])
    N = float(model["N"])
    dt = float(model.get("dt", 1.0))
    bm = float(priors.get("beta_mu", math.log(0.5)))
    bs = float(priors.get("beta_sigma", 1.0))
    gm = float(priors.get("gamma_mu", math.log(0.2)))
    gs = float(priors.get("gamma_sigma", 1.0))
    if bs <= 0 or gs <= 0:
        raise ValueError("likemc: prior sigmas must be positive")
    it = int(n_iter)
    if it < 1:
        raise ValueError("likemc: need at least one iteration")
    if float(step) <= 0:
        raise ValueError("likemc: the proposal step must be positive")

    rng = np.random.default_rng(int(seed))

    def unif():
        return rng.random()

    def norm():
        # Box-Muller from two uniforms, so the R side reproduces it
        # from the same stream in the same order.
        u1 = unif()
        while u1 <= 0.0:
            u1 = unif()
        u2 = unif()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(
            2.0 * math.pi * u2)

    def logpost(b, g):
        if b <= 0 or g <= 0:
            return float("-inf")
        lam = sir_incidence(b, g, S0, I0, N, len(y), dt)
        return (poisson_loglik(y, lam) + _lognorm_lpdf(b, bm, bs)
                + _lognorm_lpdf(g, gm, gs))

    b, g = math.exp(bm), math.exp(gm)
    lp = logpost(b, g)
    chain, n_acc = [], 0
    for _ in range(it):
        pb = b * math.exp(float(step) * norm())
        pg = g * math.exp(float(step) * norm())
        plp = logpost(pb, pg)
        if math.log(unif()) < plp - lp:
            b, g, lp = pb, pg, plp
            n_acc += 1
        chain.append([b, g, lp])
    kept = chain[int(burn):]
    if not kept:
        raise ValueError("likemc: the burn-in consumed the whole chain")
    nb = len(kept)
    mb = sum(r[0] for r in kept) / nb
    mg = sum(r[1] for r in kept) / nb
    r0 = [r[0] / r[1] for r in kept]
    sr = sorted(r0)

    def q(p):
        i = min(nb - 1, max(0, int(p * (nb - 1))))
        return sr[i]

    return RichResult(payload={
        "estimate": [mb, mg], "beta_mean": mb, "gamma_mean": mg,
        "chain": kept, "n_draws": nb, "n_iter": it,
        "acceptance_rate": n_acc / float(it),
        "R0_mean": sum(r0) / nb,
        "R0_q025": q(0.025), "R0_median": q(0.5), "R0_q975": q(0.975),
        "logpost_final": lp, "seed": int(seed), "step": float(step),
        "method": "random-walk Metropolis on (log beta, log gamma) "
                  "with a Poisson SIR incidence likelihood "
                  "(O'Neill & Roberts 1999)",
    })


def cheatsheet():
    return ("likemc: Poisson likelihood on deterministic SIR "
            "incidence, sampled by random-walk Metropolis on "
            "(log beta, log gamma). The data identify R0 = beta/gamma "
            "far better than either rate alone, which is why the "
            "posterior is sampled rather than maximised. Check the "
            "acceptance rate before believing the chain.")
