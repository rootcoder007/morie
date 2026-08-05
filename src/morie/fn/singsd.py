# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Singular spectrum analysis of a univariate series."""

from __future__ import annotations

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["singular_spectrum"]


def singular_spectrum(y, window):
    r"""Singular spectrum analysis: decompose a series into components by
    diagonalising its lag-covariance matrix, then reconstruct each one.

    With window (embedding dimension) M and series length N, Vautard, Yiou &
    Ghil use the Toeplitz estimate of the lag-covariance matrix,

    .. math::

        c_j = \frac{1}{N - j} \sum_{t=1}^{N-j} \tilde y_t\, \tilde y_{t+j},
        \qquad C_{ij} = c_{|i-j|},

    on the centred series :math:`\tilde y = y - \bar y`.  This is the
    estimator they recommend over the trajectory-matrix (Broomhead-King)
    form because it is unbiased term by term and keeps C Toeplitz, so its
    eigenvectors stay close to sines and cosines for a quasi-periodic
    signal.  C is symmetric and is diagonalised by Jacobi rotations,
    :math:`C = E \Lambda E^{\mathsf T}`, eigenvalues sorted decreasing.

    The k-th principal component is the projection of the delay vectors on
    eigenvector :math:`E_k`,

    .. math::  a^k_t = \sum_{j=1}^{M} \tilde y_{t+j-1} E_{jk},
        \qquad t = 1, \dots, N - M + 1,

    and the reconstructed component :math:`R^k` is recovered by diagonal
    averaging of :math:`a^k_t E_{jk}` over the delay index, using the
    boundary-corrected normalisation of their eq. for :math:`R^k_t`.  The
    reconstructed components sum exactly to the centred series, which is
    reported as ``reconstruction_error``.

    Because :math:`\mathrm{tr}\,C = M c_0 = M \widehat{\mathrm{var}}(y)`,
    the eigenvalues sum to M times the series variance; ``trace_check``
    reports that identity.  The Toeplitz estimate is not
    guaranteed positive semi-definite -- each :math:`c_j` carries its own
    divisor :math:`N - j` -- so a few trailing eigenvalues may come out
    slightly negative and an individual variance fraction may exceed 1.
    That is a property of the estimator Vautard, Yiou & Ghil chose, not an
    error.

    Parameters
    ----------
    y : array-like
        Series, length N; needs N > window.
    window : int
        Embedding dimension M, at least 2 and at most N - 1.

    Returns
    -------
    RichResult
        ``estimate`` is the fraction of variance carried by the leading
        eigenvalue.  ``reconstructed`` is the leading reconstructed
        component, ``eigenvalues`` the full spectrum.

    References
    ----------
    Vautard, R., Yiou, P. & Ghil, M. (1992). Singular-spectrum analysis: a
    toolkit for short, noisy chaotic signals. Physica D 58(1-4), 95-126.
    doi:10.1016/0167-2789(92)90103-T
    """
    x = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float)).tolist()]
    N = len(x)
    M = int(window)
    if N < 3:
        raise ValueError("singular_spectrum: y must have at least 3 points")
    if M < 2 or M > N - 1:
        raise ValueError("singular_spectrum: window must satisfy 2 <= window <= len(y) - 1")

    mu = sum(x) / N
    z = [v - mu for v in x]

    c = [0.0] * M
    for j in range(M):
        s = 0.0
        for t in range(N - j):
            s += z[t] * z[t + j]
        c[j] = s / (N - j)

    C = [[c[abs(i - j)] for j in range(M)] for i in range(M)]
    vals, vecs = core.jacobi(C)
    # jacobi returns eigenvalues ascending with each eigenvector sign-fixed
    # (largest-magnitude entry positive), so reversing gives them descending
    # with no sort and no tie-break for the two arms to disagree about.
    order = list(range(M - 1, -1, -1))
    lam = [vals[k] for k in order]
    E = [[vecs[i][k] for k in order] for i in range(M)]

    K = N - M + 1
    A = [[0.0] * M for _ in range(K)]
    for t in range(K):
        for k in range(M):
            s = 0.0
            for j in range(M):
                s += z[t + j] * E[j][k]
            A[t][k] = s

    # diagonal averaging with the Vautard-Yiou-Ghil boundary normalisation
    def reconstruct(k):
        R = [0.0] * N
        for t in range(N):
            lo = max(0, t - K + 1)
            hi = min(M - 1, t)
            s = 0.0
            cnt = 0
            for j in range(lo, hi + 1):
                s += A[t - j][k] * E[j][k]
                cnt += 1
            R[t] = s / cnt
        return R

    R1 = reconstruct(0)
    total = sum(lam)
    frac = [(v / total) if total > 0.0 else float("nan") for v in lam]

    # full reconstruction must return the centred series exactly
    full = [0.0] * N
    for k in range(M):
        Rk = reconstruct(k)
        for t in range(N):
            full[t] += Rk[t]
    rec_err = max(abs(full[t] - z[t]) for t in range(N))

    return RichResult(
        payload={
            "estimate": frac[0],
            "eigenvalues": lam,
            "variance_fraction": frac,
            "reconstructed": R1,
            "leading_fraction": frac[0],
            "pair_fraction": (frac[0] + frac[1]) if M > 1 else frac[0],
            "total_variance": total,
            "trace_check": abs(total - M * c[0]),
            "reconstruction_error": rec_err,
            "c0": c[0],
            "mean": mu,
            "n": float(N),
            "window": float(M),
            "n_lagged": float(K),
            "method": "Singular spectrum analysis, Toeplitz lag covariance (Vautard, Yiou & Ghil 1992)",
        }
    )


def cheatsheet():
    return "singsd: singular spectrum analysis (SSA) of a univariate series"


# compact alias per ledger/NAMING.md
singularspectrum = singular_spectrum
