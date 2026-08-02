# morie.fn -- function file (rootcoder007/morie)
"""ARE by the Pitman efficacy ratio."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_are_def"]


def gibbons_are_def(efficacy_T, efficacy_T_star, n=None):
    r"""Theorem 13.2.1: for tests satisfying the Pitman regularity
    conditions against the same local alternative sequence, the
    asymptotic relative efficiency is the ratio of squared
    efficacies:

    .. math:: \mathrm{ARE}(T, T^*) =
              \left[\frac{c(T)}{c(T^*)}\right]^2
              = \lim \frac{n^*}{n},

    the limiting sample-size ratio for equal power. The efficacy
    :math:`c(T) = \mu'(\theta_0) / (\sqrt n \, \sigma)` is what each
    test must supply; this module does the ratio and its
    interpretation.

    Parameters
    ----------
    efficacy_T, efficacy_T_star : float > 0
        Efficacies of the two tests at the null.
    n : int, optional
        A sample size for T; the equal-power size for T* is then
        n * ARE.

    Returns
    -------
    RichResult
        keys: ``are``, ``sample_size_ratio`` (n*/n = ARE),
        ``n_star_for_equal_power`` (if n given), ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 13.2.1.
    """
    cT = float(efficacy_T)
    cS = float(efficacy_T_star)
    if cT <= 0 or cS <= 0:
        raise ValueError("efficacies must be positive.")
    are = (cT / cS) ** 2
    payload = {
        "are": float(are), "sample_size_ratio": float(are),
        "method": "ARE = [c(T)/c(T*)]^2 = lim n*/n (Gibbons Theorem 13.2.1)",
    }
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        payload["n_star_for_equal_power"] = float(n * are)
    return RichResult(payload=payload)


def cheatsheet():
    return "gb1321: ARE = squared efficacy ratio = limiting sample-size ratio"
