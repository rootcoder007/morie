# morie.fn -- function file (rootcoder007/morie)
"""The periodogram, Sec. 4.7.1."""

from math import cos, fsum, pi, sin

from ._richresult import RichResult
from ._spx import mean, vec

__all__ = [
    "spectral_density",
    "pgram",
]


def spectral_density(y):
    """Periodogram of a one-dimensional lattice process, Sec. 4.7.1.1.

    Schabenberger & Gotway (2005) define the periodogram on a rectangular
    r x c lattice at eq (4.57),

        I(w1,w2) = |sum_u sum_v Z(u,v) exp{-i(w1 u + w2 v)}|^2
                   / {(2 pi)^2 r c},

    and specialise to R^1 in Sec. 4.7.1.1, where the Fourier frequencies
    are w_j = 2 pi j / r and

        2 pi I(w_j) = Chat(0) + 2 sum_{k=1}^{r-1} cos(w_j k) Chat(k)  (4.58)

    with Chat(k) = r^-1 sum_{u=k+1}^{r} (Z(u-k) - Zbar)(Z(u) - Zbar).

    Both forms are computed and the largest absolute discrepancy is
    returned as ``max_difference``. That is not decoration: the book's
    point in Sec. 4.7.1 is that the periodogram IS the Fourier transform
    of the SAMPLE covariance function, and the identity closes only when
    the 2 pi constants are carried. A large ``max_difference`` means a
    normalisation has been dropped.

    THE ZERO FREQUENCY IS EXCLUDED. The book's derivation of eq (4.58)
    turns on ``sum_u cos(w_j u) = 0``, which holds at every Fourier
    frequency EXCEPT w = 0; there the raw and mean-removed transforms
    differ by r Zbar and the identity fails. Including w = 0 is the
    standard way to make this function look broken.

    The generated stub claimed ``|FFT(y)|^2/(2 pi N)`` and returned the
    mean of `y`. The (2 pi) of eq (4.57) is squared in R^2 and single in
    R^1; the stub's spelling matches neither.

    Parameters
    ----------
    y : (r,) array-like
        Values on a one-dimensional lattice, in order.

    Returns
    -------
    RichResult
        ``omega``, ``periodogram``, ``from_covariance``,
        ``max_difference``, ``acov``, ``n``, ``method``.
    """
    z = vec(y, "y")
    r = len(z)
    if r < 4:
        raise ValueError("at least 4 lattice sites are needed")
    m = mean(z)
    d = [t - m for t in z]
    if fsum([t * t for t in d]) <= 0:
        raise ValueError("`y` is constant; the periodogram is identically 0")

    lo = -((r - 1) // 2)
    hi = r // 2
    js = [j for j in range(lo, hi + 1) if j != 0]
    omega = [2.0 * pi * j / r for j in js]

    direct = []
    for w in omega:
        re = fsum([d[u] * cos(w * (u + 1)) for u in range(r)])
        im = fsum([-d[u] * sin(w * (u + 1)) for u in range(r)])
        direct.append((re * re + im * im) / (2.0 * pi * r))

    acov = [fsum([d[u - k] * d[u] for u in range(k, r)]) / r
            for k in range(r)]
    viacov = []
    for w in omega:
        s = acov[0] + 2.0 * fsum([cos(w * k) * acov[k] for k in range(1, r)])
        viacov.append(s / (2.0 * pi))

    gap = max([abs(direct[i] - viacov[i]) for i in range(len(omega))])

    return RichResult(payload={
        "omega": omega,
        "periodogram": direct,
        "from_covariance": viacov,
        "max_difference": gap,
        "acov": acov,
        "zero_frequency_excluded": True,
        "n": r,
        "method": ("Periodogram, Schabenberger & Gotway (2005) eq (4.57) "
                   "specialised to R^1 in Sec. 4.7.1.1, checked against "
                   "eq (4.58)"),
    })


def cheatsheet():
    return "specdn: periodogram, eqs (4.57)-(4.58)"


# compact alias per ledger/NAMING.md
pgram = spectral_density
