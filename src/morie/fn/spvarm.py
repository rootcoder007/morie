# morie.fn -- function file (rootcoder007/morie)
"""The spherical semivariogram, eqs (4.14)-(4.15)."""

from ._richresult import RichResult
from ._spx import vec

__all__ = [
    "spherical_variogram_model",
    "sphvario",
]


def spherical_variogram_model(h, c0=0.0, c=1.0, a=1.0):
    """Spherical covariance and semivariogram, Schabenberger & Gotway Sec. 4.3.3.

    Sec. 4.3.3 builds the spherical family by convolving the indicator of a
    sphere of diameter alpha with white noise. The d = 3 member, eq (4.13),
    is "the" spherical model, and the book prints its covariance and
    semivariogram as

        C(h)     = sigma^2 {1 - 3h/(2 alpha) + (1/2)(h/alpha)^3}   eq (4.14)
        gamma(h) = sigma^2 {  3h/(2 alpha) - (1/2)(h/alpha)^3}     eq (4.15)

    for h <= alpha, with C(h) = 0 and gamma(h) = sigma^2 beyond. Note that
    eqs (4.14)-(4.15) carry NO nugget: the nugget is added separately, per
    Sec. 4.3.6 ("Models with Nugget Effects"), giving

        gamma(h) = 0                                for h = 0
                 = c0 + c {3h/(2a) - (1/2)(h/a)^3}  for 0 < h <= a
                 = c0 + c                           for h > a.

    gamma(0) = 0 always; the discontinuity at the origin IS the nugget and
    dropping the h = 0 case is the usual way to get it wrong. The
    correlation is exactly zero at h = a (a TRUE range, unlike the
    exponential model), which is what produces the visible kink the book
    describes at Figure 4.3.

    The generated stub cited "Cressie (1993) Sec. 2.3" and returned the mean
    of `h`. Cressie has the model too, but the equation verified here is
    Schabenberger & Gotway eq (4.15).

    Parameters
    ----------
    h : array-like
        Lag distances, non-negative.
    c0 : float
        Nugget.
    c : float
        Partial sill (the sigma^2 of eq (4.15)).
    a : float
        Range alpha; must be positive.

    Returns
    -------
    RichResult
        ``h``, ``gamma``, ``cov``, ``nugget``, ``psill``, ``sill``,
        ``range``, ``n``, ``method``.
    """
    hh = vec(h, "h")
    if any(t < 0 for t in hh):
        raise ValueError("`h` must be non-negative")
    c0 = float(c0)
    c = float(c)
    a = float(a)
    if a <= 0:
        raise ValueError("`a` (the range) must be positive")
    if c0 < 0:
        raise ValueError("`c0` (the nugget) must be non-negative")
    if c < 0:
        raise ValueError("`c` (the partial sill) must be non-negative")

    gam = []
    cov = []
    for t in hh:
        if t == 0.0:
            gam.append(0.0)
            cov.append(c0 + c)
        elif t <= a:
            u = t / a
            s = 1.5 * u - 0.5 * u * u * u
            gam.append(c0 + c * s)
            cov.append(c * (1.0 - s))
        else:
            gam.append(c0 + c)
            cov.append(0.0)

    return RichResult(payload={
        "h": hh,
        "gamma": gam,
        "cov": cov,
        "nugget": c0,
        "psill": c,
        "sill": c0 + c,
        "range": a,
        "true_range": True,
        "n": len(hh),
        "method": ("Spherical semivariogram, Schabenberger & Gotway (2005) "
                   "eq (4.15) with the nugget of Sec. 4.3.6; covariance "
                   "eq (4.14)"),
    })


def cheatsheet():
    return "spvarm: spherical semivariogram, eq (4.15)"


# compact alias per ledger/NAMING.md
sphvario = spherical_variogram_model
