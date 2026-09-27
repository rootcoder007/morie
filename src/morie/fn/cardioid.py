"""Cardioid distribution on the circle: density, distribution function, quantile and random draws.

Jeffreys, H. (1961). Theory of Probability, 3rd ed.; Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics, Sec 3.5.5.
"""

import math

from ._distcore import bisect_q, draws, out, vec
from ._richresult import RichResult

__all__ = ["cardioid"]


def cardioid(x=None, mu=0.0, rho=0.25, p=None, n=0, seed=0):
    r"""Cardioid distribution on the circle: f(t) = (1 + 2 rho cos(t - mu)) / (2 pi), 0 <= t < 2 pi, |rho| <= 1/2.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    mu : float
        Parameter.
    rho : float
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
    Jeffreys, H. (1961). Theory of Probability, 3rd ed.; Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics, Sec 3.5.5.
    Matches ``circular::dcardioid(x, mu = circular(mu), rho = rho)``.

    Examples
    --------
    >>> round(cardioid(1.0, mu=1.0, rho=0.3)["cdf"], 10)
    0.2395095031
    """
    if not (abs(rho) <= 0.5):
        raise ValueError("invalid parameters: need abs(rho) <= 0.5")

    def pdf(v):
        return (1 + 2 * rho * math.cos(v - mu)) / (2 * math.pi) if 0 <= v <= 2 * math.pi else 0.0

    def cdf(v):
        return (
            0.0
            if v <= 0
            else 1.0
            if v >= 2 * math.pi
            else (v + 2 * rho * (math.sin(v - mu) + math.sin(mu))) / (2 * math.pi)
        )

    def qf(u):
        return bisect_q(cdf, u, 0.0, 2 * math.pi)

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
        title="Cardioid distribution on the circle",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "cardioid: Cardioid distribution on the circle; pdf, cdf, quantile, Philox draws."
