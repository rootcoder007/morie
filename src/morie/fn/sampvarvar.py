"""Variance of the sample variance: Var(s^2) = (mu4 - sigma^4 (n-3)/(n-1)) / n.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.94).
"""

from ._richresult import RichResult

__all__ = ["sampvarvar"]


def sampvarvar(values, probs, n):
    """Variance of the unbiased sample variance s^2 of n i.i.d. draws from a pmf.

    With mu = sum p x, sigma^2 = sum p (x - mu)^2 and the fourth central moment
    mu4 = sum p (x - mu)^4, Var(s^2) = (mu4 - sigma^4 (n - 3)/(n - 1)) / n.

    Parameters
    ----------
    values, probs : sequences
        Outcomes and their probabilities (summing to 1).
    n : int
        Sample size, >= 2.

    Returns
    -------
    RichResult
        Keys: var_s2, sigma2, mu4.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.94).

    Examples
    --------
    >>> round(sampvarvar([1, 2, 3, 4, 5, 6], [1 / 6] * 6, 2)["var_s2"], 6)
    11.618056
    """
    x = [float(v) for v in values]
    p = [float(v) for v in probs]
    if len(x) != len(p) or not x or min(p) < 0 or abs(sum(p) - 1) > 1e-9:
        raise ValueError("values and probs must match and probs must sum to 1")
    if int(n) != n or n < 2:
        raise ValueError("n must be an integer >= 2")
    mu = sum(a * b for a, b in zip(p, x))
    s2 = sum(a * (b - mu) ** 2 for a, b in zip(p, x))
    mu4 = sum(a * (b - mu) ** 4 for a, b in zip(p, x))
    v = (mu4 - s2 * s2 * (n - 3) / (n - 1)) / n
    return RichResult(
        title="Variance of the sample variance",
        summary_lines=[("Var(s^2)", v), ("sigma^2", s2), ("mu4", mu4)],
        payload={"var_s2": v, "sigma2": s2, "mu4": mu4},
    )


def cheatsheet():
    return "sampvarvar: Var(s^2) = (mu4 - sigma^4 (n-3)/(n-1))/n. Morin (2016) eq (3.94)."
