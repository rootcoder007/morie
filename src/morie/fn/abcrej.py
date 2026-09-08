"""ABC rejection sampler (Pritchard et al. 1999)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["abcrej", "abc_rejection"]


def abcrej(sim, obs, eps, prior, n_draws=1000, seed=0):
    """
    Approximate Bayesian computation by rejection.

    Repeat for each of `n_draws` candidates: draw theta from the prior,
    simulate summary statistics s = sim(theta, rng), and accept theta
    iff ||s - obs||_2 <= eps. The accepted draws are (approximately)
    from p(theta | ||S - s_obs|| <= eps), which converges to the
    posterior as eps -> 0. This is the Pritchard et al. scheme
    (simulate from the prior, keep parameters whose simulated summaries
    fall within a tolerance of the observed summaries), stated as
    Algorithm "ABC rejection sampling" in the overview source.

    Sources
    -------
    Pritchard, J. K., Seielstad, M. T., Perez-Lezaun, A. & Feldman,
    M. W. (1999). Population growth of human Y chromosomes: a study of
    Y chromosome microsatellites. *Molecular Biology and Evolution*,
    16(12), 1791-1798 (the rejection-ABC scheme; existence and content
    verified via the journal listing).
    Sisson, S. A., Fan, Y. & Beaumont, M. A. (2018). Overview of
    Approximate Bayesian Computation. arXiv:1802.09720, Sec. 1.5
    ("rejection sampling ABC" algorithm)
    (fetched-wave3/sisson-2018-abc-overview.pdf).
    Ghosal, S. & van der Vaart, A. (2017). *Fundamentals of
    Nonparametric Bayesian Inference*, Cambridge UP (ABC discussion,
    MCMC appendix; local PDF: WD_BLACK/library/pdf/Fundamentals of
    Nonparametric Bayesian Inference).

    Parameters
    ----------
    sim : callable
        sim(theta, rng) -> summary-statistic vector (list/array).
    obs : array-like
        Observed summary statistics.
    eps : float
        Acceptance tolerance on the Euclidean distance.
    prior : list of (low, high) pairs
        Independent uniform prior box for each coordinate of theta.
    n_draws : int
        Number of prior draws.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: samples (accepted thetas), n_accepted, acceptance_rate,
        distances (accepted), posterior_mean.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be positive")
    bounds = [(float(lo), float(hi)) for lo, hi in prior]
    if any(hi <= lo for lo, hi in bounds):
        raise ValueError("each prior pair must satisfy low < high")
    rng = np.random.default_rng(seed)
    accepted = []
    dists = []
    for _ in range(int(n_draws)):
        theta = [rng.uniform(lo, hi) for lo, hi in bounds]
        s = np.asarray(sim(theta, rng), dtype=float).ravel()
        if s.size != obs.size:
            raise ValueError("sim() must return summaries matching obs")
        d = float(np.sqrt(np.sum((s - obs) ** 2)))
        if d <= eps:
            accepted.append([float(t) for t in theta])
            dists.append(d)
    k = len(accepted)
    if k:
        p = len(bounds)
        pm = [sum(a[j] for a in accepted) / k for j in range(p)]
    else:
        pm = [np.nan for _ in bounds]
    return RichResult(payload={
        "samples": accepted, "n_accepted": int(k),
        "acceptance_rate": k / float(n_draws), "distances": dists,
        "posterior_mean": pm, "eps": eps, "n_draws": int(n_draws),
        "seed": int(seed),
        "method": "ABC rejection (Pritchard et al. 1999)",
    })


# long descriptive alias (stub-era name)
abc_rejection = abcrej


def cheatsheet():
    return "abcrej: prior draws kept iff ||summary(sim) - summary(obs)|| <= eps"
