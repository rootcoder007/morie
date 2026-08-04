"""Running trace summaries for MCMC chains."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["trace_plot"]


def _quantile7(sorted_vals, p):
    """Type-7 sample quantile, the default of R's ``quantile``."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    h = (n - 1) * p
    lo = int(h)
    if lo >= n - 1:
        return sorted_vals[n - 1]
    frac = h - lo
    return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])


def trace_plot(chains, probs=(0.025, 0.5, 0.975)):
    """
    MCMC trace summaries: running mean and running quantile bands

    Formula: running mean m_t = (1/t) sum_{s<=t} x_s, with bands the
    running sample quantiles of x_1, ..., x_t

    The quantities a trace plot is read for, computed rather than drawn.
    For each chain and each iteration ``t`` the cumulative statistics of
    the draws so far are reported: the running mean, and the running
    sample quantiles at ``probs``.  A chain that has converged shows
    these settling to horizontal lines; a chain that has not shows them
    still drifting.  Comparing the lines across chains is the visual form
    of the between-chain comparison that Gelman-Rubin makes numerically.

    The running quantiles are the cumulative quantiles that
    ``coda::cumuplot`` draws, and are computed with the type-7 definition
    that is the default of R's ``quantile``, so that the two arms and
    coda agree.

    Because the statistics are cumulative this is O(n log n) per chain
    per probability and does no smoothing: nothing is thinned, no burn-in
    is discarded, and the first few iterations are reported as they are
    even though a quantile of one or two draws is nearly meaningless.

    Parameters
    ----------
    chains : array-like
        Either a 1-D sequence of draws (a single chain) or a 2-D
        array-like whose columns are chains and whose rows are
        iterations.
    probs : sequence of float
        Probabilities at which the running bands are reported.  Each must
        lie in [0, 1].

    Returns
    -------
    result : RichResult
        Keys: running_mean, bands, probs, n_chains, n_iter, final_mean,
        method.

        ``running_mean`` is a list with one entry per chain, each a list
        of length ``n_iter``.  ``bands`` is a list with one entry per
        chain, each a list with one entry per probability, each of those
        a list of length ``n_iter``.

    References
    ----------
    Plummer M, Best N, Cowles K & Vines K (2006).  CODA: convergence
    diagnosis and output analysis for MCMC.  R News 6(1), 7-11.  The
    running-quantile trace is the ``cumuplot`` diagnostic of that
    package.
    """
    arr = np.asarray(chains, dtype=float)
    lst = arr.tolist()
    if len(lst) == 0:
        raise ValueError("chains must be non-empty")
    if not isinstance(lst[0], list):
        cols = [[float(v) for v in lst]]
    else:
        n_iter_in = len(lst)
        n_ch = len(lst[0])
        cols = []
        for j in range(n_ch):
            cols.append([float(lst[i][j]) for i in range(n_iter_in)])
    n_iter = len(cols[0])
    if n_iter == 0:
        raise ValueError("chains must have at least one iteration")
    pv = [float(p) for p in probs]
    for p in pv:
        if not (0.0 <= p <= 1.0):
            raise ValueError("probs must lie in [0, 1]")

    running_mean = []
    bands = []
    for col in cols:
        run = []
        total = 0.0
        for t in range(n_iter):
            total += col[t]
            run.append(total / (t + 1))
        running_mean.append(run)
        per_prob = [[] for _ in pv]
        seen = []
        for t in range(n_iter):
            # insertion into the sorted prefix keeps this O(n^2) in the
            # worst case but avoids re-sorting the prefix at every step
            v = col[t]
            lo, hi = 0, len(seen)
            while lo < hi:
                mid = (lo + hi) // 2
                if seen[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            seen.insert(lo, v)
            for k, p in enumerate(pv):
                per_prob[k].append(_quantile7(seen, p))
        bands.append(per_prob)

    final_mean = [run[n_iter - 1] for run in running_mean]
    return RichResult(
        payload={
            "running_mean": running_mean,
            "bands": bands,
            "probs": pv,
            "n_chains": len(cols),
            "n_iter": n_iter,
            "final_mean": final_mean,
            "method": "MCMC running trace summaries (coda cumuplot)",
        }
    )


def cheatsheet():
    return "baytrace: MCMC running mean and running quantile bands"


# compact alias per ledger/NAMING.md
traceplot = trace_plot
