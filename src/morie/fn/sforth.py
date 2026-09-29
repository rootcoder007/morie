# morie.fn -- function file (rootcoder007/morie)
"""Orthogonality check of selected eigenvectors."""

import math

from ._containers import SpatialResult
from ._qpcore import ssum


def sforth(evecs, tol=1e-8):
    r"""Orthonormality of selected Moran eigenvectors and their orthogonality to the constant.

    Moran eigenvectors of ``M W M`` (``M = I - 11'/n``) are mutually orthogonal,
    of unit length and orthogonal to the intercept (Griffith 2003), which is
    what lets a filter's coefficients be read one at a time. For the ``n x
    k`` matrix ``E`` (columns = eigenvectors) the statistic is ``max |E'E -
    I|``; ``extra`` holds ``max |1'E| / sqrt(n)`` and ``orthonormal`` (both
    below ``tol``).

    References
    ----------
    Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*.
    Springer.

    Examples
    --------
    >>> s = 1 / math.sqrt(2)
    >>> r = sforth([[s, 0.5], [0.0, -0.5], [-s, 0.5], [0.0, -0.5]])
    >>> round(r.statistic, 12), r.extra["orthonormal"]
    (0.0, True)
    """
    E = [[float(v) for v in r] for r in (evecs.tolist() if hasattr(evecs, "tolist") else evecs)]
    n, k = len(E), len(E[0])
    dev = max(
        abs(ssum(E[i][a] * E[i][b] for i in range(n)) - (1.0 if a == b else 0.0)) for a in range(k) for b in range(k)
    )
    const = max(abs(ssum(E[i][a] for i in range(n))) for a in range(k)) / math.sqrt(n)
    return SpatialResult(
        name="sforth", statistic=dev, extra={"max_constant_projection": const, "orthonormal": dev < tol and const < tol}
    )


sforth_fn = sforth


def cheatsheet() -> str:
    return "sforth(evecs) -> max |E'E - I| and |1'E|/sqrt(n) of selected eigenvectors."
