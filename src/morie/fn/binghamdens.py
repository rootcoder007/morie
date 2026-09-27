"""Bingham distribution on the circle and the sphere.

Bingham, C. (1974). An antipodally symmetric distribution on the sphere. Annals of Statistics 2, 1201-1225.
"""

import math

from . import _array_core as np
from ._distcore import gauss_legendre
from ._mvcore import as_matrix, log_bessel_i, points
from ._richresult import RichResult


def _ssum(it):
    # plain left-to-right summation: sum() of floats is compensated from Python 3.12 on, which
    # would make results depend on the Python version and differ from the R arm
    s = 0.0
    for v in it:
        s += v
    return s


__all__ = ["binghamdens"]


def binghamdens(x, A):
    r"""f(x) = exp(x' A x) / c(A) for unit vectors x in dimension p = 2 or 3 (surface measure).

    With eigenvalues l_1 >= ... of the symmetric A:
    p = 2: c(A) = 2 pi exp((l_1 + l_2)/2) I_0((l_1 - l_2)/2);
    p = 3: c(A) = 2 pi integral_{-1}^{1} exp(l_3 z^2 + (l_1 + l_2)(1 - z^2)/2) I_0((l_1 - l_2)(1 - z^2)/2) dz,
    the azimuth integrated analytically and the remaining smooth integral by
    composite 16-point Gauss-Legendre. Higher dimensions are not supported.

    Parameters
    ----------
    x : unit vector or list of unit vectors
    A : p x p symmetric matrix

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, log_normalizer.

    References
    ----------
    Bingham, C. (1974). Annals of Statistics 2, 1201-1225.
    Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics, Sec 9.4.3.

    Examples
    --------
    >>> round(binghamdens([1.0, 0.0, 0.0], [[0.0] * 3] * 3)["pdf"] * 4 * math.pi, 12)
    1.0
    """
    M = as_matrix(A)
    d = len(M)
    if d not in (2, 3) or any(abs(M[i][j] - M[j][i]) > 1e-12 for i in range(d) for j in range(d)):
        raise ValueError("A must be a symmetric 2 x 2 or 3 x 3 matrix")
    lam = sorted(np.linalg.eigh(np.asarray(M))[0].tolist(), reverse=True)

    def li0(z):
        return log_bessel_i(0, abs(z)) if z != 0 else 0.0

    if d == 2:
        logc = math.log(2 * math.pi) + (lam[0] + lam[1]) / 2 + li0((lam[0] - lam[1]) / 2)
    else:
        l1, l2, l3 = lam
        shift = max(l1, l3)
        f = lambda z: math.exp(l3 * z * z + (l1 + l2) * (1 - z * z) / 2 + li0((l1 - l2) * (1 - z * z) / 2) - shift)  # noqa: E731
        logc = (
            math.log(2 * math.pi)
            + shift
            + math.log(gauss_legendre(f, -1.0, 1.0, n=max(64, int(8 * (abs(l1) + abs(l3))))))
        )
    pts, single = points(x)
    lp = []
    for v in pts:
        if len(v) != d or abs(_ssum(t * t for t in v) - 1) > 1e-9:
            raise ValueError("x must be unit vectors of the dimension of A")
        lp.append(_ssum(v[i] * M[i][j] * v[j] for i in range(d) for j in range(d)) - logc)
    return RichResult(
        title="Bingham distribution",
        summary_lines=[("dimension", d)],
        payload={
            "pdf": math.exp(lp[0]) if single else [math.exp(q) for q in lp],
            "logpdf": lp[0] if single else lp,
            "log_normalizer": logc,
        },
    )


def cheatsheet():
    return "binghamdens: Bingham density on the circle or sphere with exact/quadrature normalizing constant."
