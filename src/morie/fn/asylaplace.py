"""Asymmetric Laplace distribution: density, distribution function, quantile and random draws.

Kotz, S., Kozubowski, T. J. & Podgorski, K. (2001). The Laplace Distribution and Generalizations. Birkhauser, ch. 3.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["asylaplace"]


def asylaplace(x=None, loc=0.0, scale=1.0, kappa=1.0, p=None, n=0, seed=0):
    r"""Asymmetric Laplace distribution: f(x) = (sqrt2/scale) kappa/(1 + kappa^2) exp(-sqrt2 kappa (x - loc)/scale) for x >= loc, exp(-sqrt2 (loc - x)/(scale kappa)) below.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    loc : float
        Parameter.
    scale : float
        Parameter.
    kappa : float
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
    Kotz, S., Kozubowski, T. J. & Podgorski, K. (2001). The Laplace Distribution and Generalizations. Birkhauser, ch. 3.
    Matches ``VGAM::dalap(x, location = loc, scale = scale, kappa = kappa)``.

    Examples
    --------
    >>> round(asylaplace(0.1, loc=0.2, scale=1.1, kappa=1.7)["cdf"], 10)
    0.6888174108
    """
    if not (scale > 0 and kappa > 0):
        raise ValueError("invalid parameters: need scale > 0 and kappa > 0")

    def pdf(v):
        return (
            (math.sqrt(2) / scale)
            * kappa
            / (1 + kappa * kappa)
            * (
                math.exp(-math.sqrt(2) * kappa / scale * (v - loc))
                if v >= loc
                else math.exp(-math.sqrt(2) / (scale * kappa) * (loc - v))
            )
        )

    def cdf(v):
        return (
            kappa * kappa / (1 + kappa * kappa) * math.exp(-math.sqrt(2) / (scale * kappa) * (loc - v))
            if v < loc
            else 1 - math.exp(-math.sqrt(2) * kappa / scale * (v - loc)) / (1 + kappa * kappa)
        )

    def qf(u):
        return (
            loc + scale * kappa / math.sqrt(2) * math.log(u * (1 + kappa * kappa) / (kappa * kappa))
            if u <= kappa * kappa / (1 + kappa * kappa)
            else loc - scale / (math.sqrt(2) * kappa) * math.log((1 - u) * (1 + kappa * kappa))
        )

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
        title="Asymmetric Laplace distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "asylaplace: Asymmetric Laplace distribution; pdf, cdf, quantile, Philox draws."
