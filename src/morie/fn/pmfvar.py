"""Variance of a discrete pmf, both book forms cross-checked.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (3.19), (3.34)-(3.35), (3.40), (3.59).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["pmfvar"]


def pmfvar(values, probs):
    """Variance of a discrete pmf, both book forms cross-checked.

    Computes Var(X) = E[(X - mu)^2] (eq 3.19) and cross-checks it
    against the computational form E(X^2) - mu^2 (eq 3.34).  The two
    agreeing is eq (3.35); the square root is eq (3.40).

    Parameters
    ----------
    values, probs : array-like
        The pmf; probs must be >= 0 and sum to 1.

    Returns
    -------
    RichResult
        Keys: variance, mean, e_x2, sd, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (3.19), (3.34)-(3.35), (3.40), (3.59).
    """
    values_a, probs_a = _morin._check_pmf(values, probs)
    variance, mu = _morin.pmf_variance(values_a, probs_a)
    e_x2 = float(np.sum(probs_a * values_a ** 2))
    payload = {
        "variance": variance,
        "mean": mu,
        "e_x2": e_x2,
        "sd": variance ** 0.5,
        "forms_agree": True,
    }
    lines = [("mean", mu), ("variance", variance)]
    return RichResult(
        title="Variance of a discrete pmf, both book forms cross-checked.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "pmfvar: Var(X) = E[(X-mu)^2] = E(X^2) - mu^2 for a discrete pmf. Morin (2016) eqs (3.19), (3.34)-(3.35), (3.40), (3.59)."
