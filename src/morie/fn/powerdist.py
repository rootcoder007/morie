"""Power function distribution: density, distribution function, quantile and random draws.

Johnson, Kotz & Balakrishnan (1995). Continuous Univariate Distributions, Vol. 2, ch. 25.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["powerdist"]


def powerdist(x=None, alpha=1.0, beta=1.0, p=None, n=0, seed=0):
    r"""Power function distribution: F(x) = (x/alpha)^beta, 0 < x < alpha.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    alpha : float
        Parameter.
    beta : float
        Parameter.
    p : float or sequence, optional
        Probabilities for the quantile function.
    n : int
        Number of random draws.
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, cdf (when x is given), quantile (when p is given), random.

    References
    ----------
    Johnson, Kotz & Balakrishnan (1995). Continuous Univariate Distributions, Vol. 2, ch. 25.
    Matches ``extraDistr::dpower(x, alpha = alpha, beta = beta)``.

    Examples
    --------
    >>> round(powerdist(1.0, alpha=2.0, beta=3.0)["cdf"], 10)
    0.125
    """
    if not (alpha > 0 and beta > 0):
        raise ValueError("invalid parameters: need alpha > 0 and beta > 0")

    def pdf(v):
        return beta * v ** (beta - 1) / alpha**beta if 0 < v < alpha else 0.0

    def cdf(v):
        return 0.0 if v <= 0 else 1.0 if v >= alpha else (v / alpha) ** beta

    def qf(u):
        return alpha * u ** (1 / beta)

    payload = {}
    if x is not None:
        xs, sc = vec(x)
        dens = [pdf(v) for v in xs]
        payload["pdf"] = out(dens, sc)
        payload["logpdf"] = out([math.log(v) if v > 0 else -math.inf for v in dens], sc)
        payload["cdf"] = out([cdf(v) for v in xs], sc)
    if p is not None:
        ps, sc = vec(p)
        payload["quantile"] = out([qf(u) for u in ps], sc)
    if n:
        payload["random"] = draws(qf, n, seed)
    return RichResult(
        title="Power function distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "powerdist: Power function distribution; pdf, cdf, quantile, Philox draws."
