"""Nakagami distribution: density, distribution function, quantile and random draws.

Nakagami, M. (1960). The m-distribution. In Statistical Methods in Radio Wave Propagation, 3-36.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult
from ._rrng_core import pgamma, qgamma

__all__ = ["nakagami"]


def nakagami(x=None, shape=None, scale=1.0, p=None, n=0, seed=0):
    r"""Nakagami distribution: f(x) = 2 m^m x^(2m-1) exp(-m x^2/Omega) / (Gamma(m) Omega^m); F(x) = P(m, m x^2/Omega).

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
    Nakagami, M. (1960). The m-distribution. In Statistical Methods in Radio Wave Propagation, 3-36.
    Matches ``VGAM::dnaka(x, scale = scale, shape = shape)``.

    Examples
    --------
    >>> round(nakagami(1.0, shape=1.8, scale=2.5)["cdf"], 10)
    0.2112331573
    """
    if shape is None:
        raise ValueError("shape is required")
    if not (shape >= 0.5 and scale > 0):
        raise ValueError("invalid parameters: need shape >= 0.5 and scale > 0")

    def pdf(v):
        return (
            0.0
            if v <= 0
            else math.exp(
                math.log(2)
                + shape * math.log(shape)
                - math.lgamma(shape)
                - shape * math.log(scale)
                + (2 * shape - 1) * math.log(v)
                - shape * v * v / scale
            )
        )

    def cdf(v):
        return 0.0 if v <= 0 else pgamma(shape * v * v / scale, shape)

    def qf(u):
        return math.sqrt(scale * qgamma(u, shape) / shape)

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
        title="Nakagami distribution",
        summary_lines=[(k, v) for k, v in payload.items() if not isinstance(v, list)],
        payload=payload,
    )


def cheatsheet():
    return "nakagami: Nakagami distribution; pdf, cdf, quantile, Philox draws."
