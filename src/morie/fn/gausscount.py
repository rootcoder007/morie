"""Expected count of a Gaussian outcome over N repetitions.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (5.28).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["gausscount"]


def gausscount(x, n_reps=100000, mu=35.0, sigma=5.4):
    """Expected count of a Gaussian outcome over N repetitions.

    N f(x) for a Gaussian f.  The defaults are the book's dice-sum
    experiment: ten dice, mean 35, sigma 5.4, repeated 100,000 times.

    Parameters
    ----------
    x : float
        Outcome value.
    n_reps : float
        Number of repetitions, > 0.
    mu, sigma : float
        Gaussian mean and standard deviation; sigma > 0.

    Returns
    -------
    RichResult
        Keys: expected_count, mu, sigma.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (5.28).
    """
    n_r = float(n_reps)
    if n_r <= 0:
        raise ValueError("n_reps must be > 0")
    value = n_r * _morin.normal_pdf(x, mu, sigma)
    payload = {"expected_count": value, "mu": float(mu), "sigma": float(sigma)}
    lines = [("expected count", value)]
    return RichResult(
        title="Expected count of a Gaussian outcome over N repetitions.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "gausscount: N f(x) expected count for a Gaussian outcome. Morin (2016) eq (5.28)."
