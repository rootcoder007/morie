# morie.fn -- function file (rootcoder007/morie)
"""Generic expectation-maximisation driver.

Source CONSULTED: Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977),
"Maximum likelihood from incomplete data via the EM algorithm (with
discussion)", *JRSS B* 39(1):1-38.  Paywalled and not obtainable here;
what is implemented is the standard published statement of the
algorithm, and in particular its defining guarantee, Theorem 1 of that
paper: every EM iteration increases the observed-data log likelihood,

    l(theta_{t+1})  >=  l(theta_t).

That inequality is not assumed -- it is CHECKED.  The driver evaluates
``log_lik`` at every iterate and reports the increments, the smallest
of them, and a ``monotone`` flag.  A negative increment means the
supplied M-step is not maximising the surrogate, and is a defect in the
caller's E/M pair rather than in the driver; the flag surfaces it
instead of hiding it.

The driver is deliberately model-free: it owns the loop and the
monotonicity audit, and the caller owns the E and M steps.  It runs a
FIXED number of iterations, since an early exit on one language arm and
not the other would silently break Python/R parity.
"""

from ._richresult import RichResult

__all__ = ["em_algorithm"]


def em_algorithm(log_lik, Q, x0, steps):
    """Run EM for a fixed number of steps and audit its monotonicity.

    Parameters
    ----------
    log_lik : callable
        ``log_lik(theta)`` returns the observed-data log likelihood.
    Q : callable
        One combined E- and M-step: ``Q(theta)`` returns the next
        iterate, i.e. the maximiser over theta' of the expected
        complete-data log likelihood given theta.
    x0 : sequence of float
        Starting parameter vector.
    steps : int
        Number of EM iterations.

    Returns
    -------
    RichResult
        Keys ``theta``, ``loglik``, ``trace``, ``increments``,
        ``min_increment``, ``monotone``, ``steps``, ``method``.
    """
    theta = [float(v) for v in x0]
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    trace = [float(log_lik(theta))]
    for _t in range(steps):
        nxt = [float(v) for v in Q(theta)]
        if len(nxt) != len(theta):
            raise ValueError("the M-step changed the parameter length")
        theta = nxt
        trace.append(float(log_lik(theta)))
    inc = [trace[i + 1] - trace[i] for i in range(len(trace) - 1)]
    mn = min(inc) if inc else 0.0
    return RichResult(
        payload={
            "theta": theta,
            "loglik": trace[-1],
            "trace": trace,
            "increments": inc,
            "min_increment": mn,
            "monotone": mn >= -1e-9,
            "steps": steps,
            "method": "EM driver with a Dempster-Laird-Rubin Theorem 1 "
                      "monotonicity audit",
        }
    )


def cheatsheet():
    return "epsig1: Expectation-maximization"


# compact alias per ledger/NAMING.md
emalgorithm = em_algorithm
