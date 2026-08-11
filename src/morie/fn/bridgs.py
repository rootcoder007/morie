"""Bridge sampling for ratios of normalizing constants (Meng-Wong 1996)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bridgs", "bridge_sampling"]


def bridgs(draws1, draws2, log_q1, log_q2, tol=1e-12, max_iter=1000):
    """
    Iterative optimal-bridge estimate of r = c1/c2.

    With draws w_1j ~ p1 (j = 1..n1) and w_2j ~ p2 (j = 1..n2),
    unnormalized densities q1, q2, l_ij = q1(w_ij)/q2(w_ij),
    s1 = n1/(n1+n2) and s2 = n2/(n1+n2), the iteration of the source,
    Sec. 4 ("Iterative Choice of alpha", derived from the optimal
    alpha of eq. 3.5):

                     (1/n2) sum_j l_2j / (s1 l_2j + s2 r_t)
        r_{t+1}  =  -----------------------------------------
                     (1/n1) sum_j   1  / (s1 l_1j + s2 r_t)

    iterated to a fixed point. The log-ratios are shifted by a common
    constant for overflow safety (the shift cancels exactly and is
    added back to the reported log r).

    Sources
    -------
    Meng, X.-L. & Wong, W. H. (1996). Simulating ratios of normalizing
    constants via a simple identity: a theoretical exploration.
    *Statistica Sinica*, 6, 831-860, eq. (3.5) and Sec. 4
    (fetched-wave3/meng-wong-1996-bridge-sampling-statsinica.pdf).

    Parameters
    ----------
    draws1 : sequence
        Draws from the (normalized) density p1 = q1/c1.
    draws2 : sequence
        Draws from p2 = q2/c2.
    log_q1, log_q2 : callable
        Log unnormalized densities.
    tol : float
        Relative fixed-point tolerance on log r.
    max_iter : int
        Iteration cap.

    Returns
    -------
    RichResult
        Keys: ratio (r = c1/c2), log_ratio, iterations, converged.
    """
    x1 = list(draws1)
    x2 = list(draws2)
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        raise ValueError("both draw sets must be non-empty")
    ll1 = [float(log_q1(x)) - float(log_q2(x)) for x in x1]
    ll2 = [float(log_q1(x)) - float(log_q2(x)) for x in x2]
    shift = max(max(ll1), max(ll2))
    l1 = [np.exp(v - shift) for v in ll1]
    l2 = [np.exp(v - shift) for v in ll2]
    s1 = n1 / float(n1 + n2)
    s2 = n2 / float(n1 + n2)
    r = 1.0
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        num = sum(v / (s1 * v + s2 * r) for v in l2) / n2
        den = sum(1.0 / (s1 * v + s2 * r) for v in l1) / n1
        r_new = num / den
        if abs(np.log(r_new) - np.log(r)) <= tol * (1.0 + abs(np.log(r_new))):
            r = r_new
            converged = True
            break
        r = r_new
    log_ratio = float(np.log(r) + shift)
    return RichResult(payload={
        "ratio": float(np.exp(log_ratio)), "log_ratio": log_ratio,
        "iterations": int(it), "converged": bool(converged),
        "n1": int(n1), "n2": int(n2),
        "method": "Meng-Wong iterative optimal bridge (eq. 3.5 / Sec. 4)",
    })


# long descriptive alias (stub-era name)
bridge_sampling = bridgs


def cheatsheet():
    return "bridgs: iterative optimal-bridge r = c1/c2 (Meng-Wong 1996)"
