# morie.fn -- function file (rootcoder007/morie)
"""Derivatives of the quantile function."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_quantile_deriv"]


def gibbons_quantile_deriv(p, f, f_prime=None, Q=None):
    r"""Theorem 2.2.1: differentiating :math:`F(Q(p)) = p`,

    .. math:: Q'(p) = \frac{1}{f(Q(p))}, \qquad
              Q''(p) = -\frac{f'(Q(p))}{f(Q(p))^3}.

    The reciprocal-density rule is what puts :math:`1/f^2` into every
    asymptotic quantile variance (see the Ch. 2.9 moments module).

    Parameters
    ----------
    p : float in (0, 1)
        Probability level.
    f : callable
        Density.
    f_prime : callable, optional
        Density derivative; numerically differenced if omitted.
    Q : callable, optional
        Quantile function; when omitted, ``f`` is treated as a scipy
        frozen distribution if it has a ``ppf``, else Q(p) must be
        supplied.

    Returns
    -------
    RichResult
        keys: ``Q_p``, ``Q_prime``, ``Q_double_prime``, ``f_at_Q``,
        ``p``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 2.2.1.
    """
    p = float(p)
    if not 0 < p < 1:
        raise ValueError(f"p must lie in (0, 1), got {p}.")
    if hasattr(f, "ppf") and hasattr(f, "pdf"):
        dist = f
        qp = float(dist.ppf(p))
        fq = float(dist.pdf(qp))
        h = 1e-6
        fpq = (float(dist.pdf(qp + h)) - float(dist.pdf(qp - h))) / (2 * h) \
            if f_prime is None else float(f_prime(qp))
    else:
        if Q is None:
            raise ValueError("supply Q when f is a bare density function.")
        qp = float(Q(p))
        fq = float(f(qp))
        h = 1e-6
        fpq = (float(f(qp + h)) - float(f(qp - h))) / (2 * h) \
            if f_prime is None else float(f_prime(qp))
    if fq <= 0:
        raise ValueError("density is zero at the quantile; Q' is undefined.")
    return RichResult(
        payload={
            "Q_p": qp, "Q_prime": 1.0 / fq, "Q_double_prime": -fpq / fq**3,
            "f_at_Q": fq, "p": p,
            "method": "Q' = 1/f(Q), Q'' = -f'(Q)/f(Q)^3 (Gibbons Theorem 2.2.1)",
        }
    )


def cheatsheet():
    return "gb221: Q' = 1/f(Q); the source of 1/f^2 in quantile variances"
