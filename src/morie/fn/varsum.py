"""Variance of a sum of independent variables: the sum of the variances.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.25), (3.30)-(3.31).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["varsum"]


def varsum(variances):
    """Variance of a sum of independent variables: the sum of the variances.

    The running partial sums make the recursive statement of eq (3.31)
    explicit: Var(X1+...+Xn) = Var(X1+...+X_{n-1}) + Var(Xn).

    Parameters
    ----------
    variances : array-like
        Per-variable variances, each >= 0.

    Returns
    -------
    RichResult
        Keys: var_sum, partial_sums.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.25), (3.30)-(3.31).
    """
    v = np.atleast_1d(np.asarray(variances, dtype=float))
    if v.size < 1 or np.any(v < 0):
        raise ValueError("variances must be non-empty and >= 0")
    running = 0.0
    steps = []
    for x in v:
        running = running + float(x)
        steps.append(running)
    direct = _morin.var_sum_independent(v)
    if abs(running - direct) > 1e-12 * max(1.0, direct):
        raise AssertionError("recursion disagrees with direct sum")
    payload = {"var_sum": running, "partial_sums": steps}
    lines = [("Var(sum)", running)]
    return RichResult(
        title="Variance of a sum of independent variables: the sum of the variances.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "varsum: Var(X1+...+Xn) = sum Var(Xi) for independent Xi. Morin (2016) eqs (3.25), (3.30)-(3.31)."
