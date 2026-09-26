"""Conditional negative definiteness of a semivariogram matrix."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["semivariogram_cnd_check"]


def semivariogram_cnd_check(gamma, tol=1e-10):
    r"""Check that a semivariogram is conditionally negative definite at given sites.

    Schabenberger & Gotway (2005, Problem 2.2, p. 79): a valid semivariogram
    satisfies

    .. math::  \sum_{i=1}^m\sum_{j=1}^m a_i a_j\gamma(s_i - s_j) \le 0

    for all sites and all real :math:`a` with :math:`\sum a_i = 0`, because
    :math:`\mathrm{Var}[\sum a_i Z(s_i)] = -\sum\sum a_i a_j\gamma(s_i - s_j)`
    when the weights sum to zero. With :math:`P = I - J/m` the projection
    onto such vectors, the condition at the given sites is that the largest
    eigenvalue of :math:`P\Gamma P` is at most zero.

    Parameters
    ----------
    gamma : array-like, (m, m)
        Symmetric matrix of :math:`\gamma(s_i - s_j)`, zero diagonal.
    tol : float
        Relative tolerance on the largest eigenvalue, scaled by
        :math:`\max|\gamma|`.

    Returns
    -------
    RichResult
        ``max_eigenvalue`` of :math:`P\Gamma P` (on the sum-zero subspace),
        ``valid`` (``max_eigenvalue <= tol * max|gamma|``), ``m``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, Problem 2.2, p. 79 and
    Sec. 4.2.
    """
    g = [[float(v) for v in row] for row in gamma]
    m = len(g)
    if m < 2 or any(len(r) != m for r in g):
        raise ValueError("`gamma` must be a square matrix of at least 2 x 2")
    rbar = [sum(r) / m for r in g]
    cbar = [sum(g[i][j] for i in range(m)) / m for j in range(m)]
    tbar = sum(rbar) / m
    pgp = [[g[i][j] - rbar[i] - cbar[j] + tbar for j in range(m)] for i in range(m)]
    pgp = [[0.5 * (pgp[i][j] + pgp[j][i]) for j in range(m)] for i in range(m)]
    ev = sorted(float(v) for v in np.linalg.eigvalsh(np.asarray(pgp)))
    # P has a null vector (the ones vector); drop the eigenvalue belonging to it
    ones_ev = min(ev, key=abs)
    ev.remove(ones_ev)
    top = ev[-1]
    scale = max(abs(v) for r in g for v in r) or 1.0
    return RichResult(
        title="Conditional negative definiteness of a semivariogram",
        summary_lines=[("m", m), ("max eigenvalue", top), ("valid", top <= tol * scale)],
        payload={"max_eigenvalue": top, "valid": top <= tol * scale, "m": m},
    )


def cheatsheet():
    return "cndchk: sum a_i a_j gamma_ij <= 0 whenever sum a = 0, via eig(P Gamma P)"
