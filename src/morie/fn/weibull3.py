"""Three-parameter Weibull distribution: density, distribution function, quantile and random draws.

Weibull, W. (1951). A statistical distribution function of wide applicability. J. Appl. Mech. 18, 293-297.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["weibull3"]


def weibull3(x=None, shape=None, scale=1.0, loc=0.0, p=None, n=0, seed=0):
    r"""Three-parameter Weibull distribution: F(x) = 1 - exp(-((x - loc)/scale)^shape), x > loc.

    Returns the density (and its log), the distribution function at ``x``, the
    quantile function at ``p`` and ``n`` draws by inversion of morie's Philox
    uniforms (``seed``; identical in the R arm).

    Parameters
    ----------
    x : float or sequence, optional
        Evaluation points.
    shape : float
        Parameter (required).
    scale : float
        Parameter.
    loc : float
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
    Weibull, W. (1951). A statistical distribution function of wide applicability. J. Appl. Mech. 18, 293-297.
    Matches ``stats::dweibull(x - loc, shape, scale)``.

    Examples
    --------
    >>> round(weibull3(1.5, shape=1.7, scale=2.0, loc=0.5)["cdf"], 10)
    0.2649274692
    """
    if shape is None:
        raise ValueError("shape is required")
    if not (shape > 0 and scale > 0):
        raise ValueError("invalid parameters: need shape > 0 and scale > 0")

    def pdf(v):
        return (
            0.0
            if v <= loc
            else shape / scale * ((v - loc) / scale) ** (shape - 1) * math.exp(-(((v - loc) / scale) ** shape))
        )

    def cdf(v):
        return 0.0 if v <= loc else 1 - math.exp(-(((v - loc) / scale) ** shape))

    def qf(u):
        return loc + scale * (-math.log(1 - u)) ** (1 / shape)

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
        title="Three-parameter Weibull distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "weibull3: Three-parameter Weibull distribution; pdf, cdf, quantile, Philox draws."
