"""Sampling-importance-resampling (SIR, Rubin 1988)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bayisr", "importance_resample"]


def bayisr(samples, log_target, log_proposal, m, seed=0):
    """
    Sampling-importance-resampling: importance-weight draws from a
    proposal, then resample with replacement proportionally to the
    weights to get approximate draws from the target.

    Given draws x_1..x_n from the proposal g, the importance weights
    are w_i = exp(log p(x_i) - log g(x_i)) (computed with the max
    log-ratio subtracted for stability; the constant cancels in the
    normalized weights). The resample draws m indices with replacement
    with P(i) = w_i / sum w. As n -> infinity the resampled points are
    distributed according to the target p; Rubin's n/m -> infinity
    refinement governs the quality of the approximation.

    The resampling uses one uniform per draw with inverse-CDF on the
    unnormalized weights, exactly mirroring the R helper .ghc_choice_p
    so both arms consume the shared SplitMix64 stream identically.

    Sources
    -------
    Rubin, D. B. (1988). Using the SIR algorithm to simulate posterior
    distributions. In *Bayesian Statistics 3* (Bernardo, DeGroot,
    Lindley, Smith, eds.), Oxford UP, 395-402 (the SIR algorithm;
    existence verified via the volume listing).
    Smith, A. F. M. & Gelfand, A. E. (1992). Bayesian statistics
    without tears: a sampling-resampling perspective. *The American
    Statistician*, 46(2), 84-88 (weighted-bootstrap statement of the
    same resampling scheme).

    Parameters
    ----------
    samples : array-like, (n,) or (n, d)
        Draws from the proposal g.
    log_target : callable
        log p(x) up to a constant, evaluated per sample.
    log_proposal : callable
        log g(x) up to a constant, evaluated per sample.
    m : int
        Resample size.
    seed : int
        Native-RNG seed (SplitMix64; mirrored by .ghc_rng in R).

    Returns
    -------
    RichResult
        Keys: resample (m draws), indices (0-based), weights
        (normalized), ess (effective sample size 1/sum(wbar^2)).
    """
    xs = list(samples)
    n = len(xs)
    if n == 0:
        raise ValueError("`samples` must be non-empty")
    m = int(m)
    if m < 1:
        raise ValueError("m must be a positive integer")
    logw = [float(log_target(x)) - float(log_proposal(x)) for x in xs]
    mx = max(logw)
    w = [np.exp(v - mx) for v in logw]
    tot = float(sum(w))
    wbar = [v / tot for v in w]
    ess = 1.0 / sum(v * v for v in wbar)
    rng = np.random.default_rng(seed)
    idx = []
    cum = []
    acc = 0.0
    for v in w:
        acc += v
        cum.append(acc)
    for _ in range(m):
        u = rng.uniform(0.0, 1.0) * tot
        j = n - 1
        for t in range(n):
            if u <= cum[t]:
                j = t
                break
        idx.append(j)
    res = [xs[j] for j in idx]
    return RichResult(payload={
        "resample": res, "indices": idx, "weights": wbar,
        "ess": float(ess), "n": int(n), "m": m, "seed": int(seed),
        "method": "SIR weighted bootstrap (Rubin 1988; Smith-Gelfand 1992)",
    })


# long descriptive alias (stub-era name)
importance_resample = bayisr


def cheatsheet():
    return "bayisr: importance weights w = p/g then resample with P(i) = w_i / sum w"
