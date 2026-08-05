"""Multinomial theorem: (x1 + ... + xk)^N = sum C(N; n1..nk) prod xi^ni.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (1.38).
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["multiexp"]


def _compositions(N, k):
    if k == 1:
        yield (N,)
        return
    for first in range(N + 1):
        for rest in _compositions(N - first, k - 1):
            yield (first,) + rest


def multiexp(xs, N):
    """Expand (x1 + x2 + ... + xk)^N over multinomial coefficients.

    Eq (1.38) sums, over every composition n1 + ... + nk = N, the term
    C(N; n1,...,nk) x1^n1 ... xk^nk with the multinomial coefficient of
    eq (1.37).  The expansion total is cross-checked against the direct
    power (sum xs)^N.

    Parameters
    ----------
    xs : array-like
        The k values x1..xk.
    N : int
        The power, N >= 0.

    Returns
    -------
    RichResult
        Keys: k, power, n_terms, expansion, direct_power, max_coefficient.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.37)-(1.38).
    """
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    if xs.size == 0:
        raise ValueError("xs must be non-empty")
    N = _morin._check_nonneg_int(N, "N")
    k = int(xs.size)
    total = 0.0
    n_terms = 0
    max_coef = 0
    for comp in _compositions(N, k):
        coef = _morin.multinomial_coefficient(list(comp))
        term = float(coef)
        for value, power in zip(xs, comp):
            term *= float(value) ** power
        total += term
        n_terms += 1
        if coef > max_coef:
            max_coef = coef
    direct = float(np.sum(xs)) ** N
    if abs(total - direct) > 1e-9 * max(1.0, abs(direct)):
        raise AssertionError("multinomial expansion does not match the direct power")
    payload = {"k": float(k), "power": float(N), "n_terms": float(n_terms),
               "expansion": float(total), "direct_power": float(direct),
               "max_coefficient": float(max_coef)}
    return RichResult(
        title="Multinomial theorem expansion.",
        summary_lines=[("terms", n_terms), ("total", total)],
        payload=payload,
    )


def cheatsheet():
    return "multiexp: (x1+...+xk)^N = sum over n1+...+nk=N of C(N;n1..nk) prod xi^ni. Morin (2016) eq (1.38)."
