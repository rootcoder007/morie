# morie.fn -- function file (rootcoder007/morie)
"""Bridge sampling estimator of a marginal likelihood."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bridge_sampling", "bridge_sampling_marginal"]


def bridge_sampling(log_p_posterior, log_q_posterior, log_p_proposal,
                    log_q_proposal, max_iter=1000, tol=1e-10):
    r"""Marginal likelihood by the optimal bridge, iterated to a fixed point.

    Meng and Wong's optimal bridge function gives

    .. math::
       \hat r_{(t+1)} = \frac{\frac{1}{n_2}\sum_{j}
         \frac{l_{2j}}{s_1 l_{2j} + s_2 \hat r_{(t)}}}
        {\frac{1}{n_1}\sum_{i}
         \frac{1}{s_1 l_{1i} + s_2 \hat r_{(t)}}},

    with :math:`l = p^*/q`, :math:`s_1 = n_1/(n_1+n_2)`. The recursion
    is needed because the optimal bridge depends on the very ratio
    being estimated.

    Bridge sampling exists because the simpler estimators fail in ways
    that are hard to notice. The harmonic mean estimator is
    notoriously unstable -- its variance is often infinite, and it
    converges to the wrong answer while looking convergent. Naive
    importance sampling needs the proposal to dominate the posterior
    everywhere, which fails in moderate dimension. Bridge sampling
    requires only OVERLAP between the two, which is a far weaker
    condition and is why it works where the others do not.

    ``overlap`` reports how much there is. When it is small the
    estimate is unreliable no matter how tight the reported error, and
    the fix is a better proposal rather than more draws.

    Parameters
    ----------
    log_p_posterior : array-like, shape (n1,)
        Unnormalised log posterior at the POSTERIOR draws.
    log_q_posterior : array-like, shape (n1,)
        Log proposal density at the same draws.
    log_p_proposal : array-like, shape (n2,)
        Unnormalised log posterior at the PROPOSAL draws.
    log_q_proposal : array-like, shape (n2,)
        Log proposal density at the proposal draws.
    max_iter, tol : int, float

    Returns
    -------
    RichResult
        ``log_marginal``, ``iterations``, ``converged``, ``overlap``,
        ``relative_se``, ``harmonic_mean_comparison``.

    References
    ----------
    Meng and Wong (1996), *Statistica Sinica* 6:831-860.
    Gronau et al. (2017), *Journal of Mathematical Psychology*
    81:80-97, for the practical recipe and the error estimate.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> lp1, lq1 = rng.normal(size=200), rng.normal(size=200)
    >>> lp2, lq2 = rng.normal(size=200), rng.normal(size=200)
    >>> out = bridge_sampling(lp1, lq1, lp2, lq2)
    >>> bool(np.isfinite(out["log_marginal"]))
    True
    """
    lp1 = np.asarray(log_p_posterior, dtype=float).ravel()
    lq1 = np.asarray(log_q_posterior, dtype=float).ravel()
    lp2 = np.asarray(log_p_proposal, dtype=float).ravel()
    lq2 = np.asarray(log_q_proposal, dtype=float).ravel()
    if lp1.size != lq1.size:
        raise ValueError("posterior-draw arrays must agree in length.")
    if lp2.size != lq2.size:
        raise ValueError("proposal-draw arrays must agree in length.")
    n1, n2 = lp1.size, lp2.size
    if n1 < 2 or n2 < 2:
        raise ValueError("need at least 2 draws from each source.")
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)

    l1 = lp1 - lq1                          # log p*/q at posterior draws
    l2 = lp2 - lq2                          # log p*/q at proposal draws
    shift = float(np.median(np.concatenate([l1, l2])))
    e1, e2 = np.exp(l1 - shift), np.exp(l2 - shift)

    logr = 0.0
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        r = np.exp(logr)
        num = np.mean(e2 / (s1 * e2 + s2 * r))
        den = np.mean(1.0 / (s1 * e1 + s2 * r))
        if not (np.isfinite(num) and np.isfinite(den)) or den <= 0 or num <= 0:
            break
        new = np.log(num) - np.log(den)
        if abs(new - logr) < tol:
            logr = new
            converged = True
            break
        logr = new
    log_marginal = float(logr + shift)

    # overlap of the two log-ratio samples, as the fraction of each
    # sample lying inside the other's range
    lo, hi = np.percentile(l1, [5, 95])
    ov = float(np.mean((l2 >= lo) & (l2 <= hi)))
    lo2, hi2 = np.percentile(l2, [5, 95])
    ov = 0.5 * (ov + float(np.mean((l1 >= lo2) & (l1 <= hi2))))

    r = np.exp(logr)
    f1 = e1 / (s1 * e1 + s2 * r)
    f2 = 1.0 / (s1 * e2 + s2 * r)
    rel = float(np.sqrt(
        np.var(f2, ddof=1) / np.mean(f2) ** 2 / n2
        + np.var(f1, ddof=1) / np.mean(f1) ** 2 / n1
    )) if np.mean(f1) > 0 and np.mean(f2) > 0 else np.nan
    hm = float(shift - np.log(np.mean(np.exp(-(l1 - shift)))) - 2 * shift) \
        if np.all(np.isfinite(l1)) else np.nan
    return RichResult(
        payload={
            "estimate": log_marginal,
            "log_marginal": log_marginal,
            "marginal": float(np.exp(log_marginal))
            if abs(log_marginal) < 700 else np.inf,
            "iterations": int(it),
            "converged": bool(converged),
            "overlap": ov,
            "overlap_note": (
                "bridge sampling needs only OVERLAP between posterior and "
                "proposal, not domination as importance sampling does; when "
                "this is small the fix is a better proposal, not more draws"
            ),
            "relative_se": rel,
            "harmonic_mean_comparison": hm,
            "harmonic_mean_note": (
                "the harmonic mean estimator shown for contrast only: its "
                "variance is often infinite and it converges to the wrong "
                "answer while appearing to converge"
            ),
            "n_posterior": int(n1),
            "n_proposal": int(n2),
            "method": "Bridge sampling marginal likelihood",
        }
    )


def cheatsheet():
    return (
        "bdgsm: iterated optimal-bridge marginal likelihood, with the "
        "overlap it actually depends on"
    )


#: Catalogue alias for :func:`bridge_sampling`.
bridge_sampling_marginal = bridge_sampling
