# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fast covariance estimation for functional data (FACE).

Xiao L, Zipunnikov V, Ruppert D, Crainiceanu C (2016), *Fast covariance
estimation for high-dimensional functional data*, Statistics and
Computing 26:409-421; the sandwich smoother is Xiao L, Li Y, Ruppert D
(2013), *JRSS B* 75(3):577-599.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["face_smooth", "bspline_basis"]

_METHOD = "FACE sandwich-smoothed covariance with FPCA"


def bspline_basis(x, n_basis=15, degree=3, lo=None, hi=None):
    """B-spline design matrix by the Cox-de Boor recursion.

    Equally spaced interior knots, with the boundary knots repeated
    ``degree + 1`` times so the basis spans the whole interval.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_basis = int(n_basis)
    degree = int(degree)
    if degree < 0:
        raise ValueError(f"degree must be non-negative; got {degree}.")
    if n_basis < degree + 1:
        raise ValueError(
            f"n_basis must be at least degree + 1 = {degree + 1}; "
            f"got {n_basis}."
        )
    a = float(np.min(x)) if lo is None else float(lo)
    b = float(np.max(x)) if hi is None else float(hi)
    if not b > a:
        raise ValueError("the evaluation range is degenerate.")
    n_interior = n_basis - degree - 1
    interior = np.linspace(a, b, n_interior + 2)[1:-1]
    knots = np.concatenate([np.full(degree + 1, a), interior,
                            np.full(degree + 1, b)])
    m = knots.size - degree - 1
    B = np.zeros((x.size, m))
    for i in range(m):
        B[:, i] = _bspline_one(x, knots, i, degree, b)
    return B


def _bspline_one(x, t, i, k, right):
    if k == 0:
        out = ((x >= t[i]) & (x < t[i + 1])).astype(float)
        # close the last interval so the right endpoint is covered
        if t[i + 1] >= right:
            out = np.where(x == right, 1.0, out)
        return out
    out = np.zeros_like(x)
    d1 = t[i + k] - t[i]
    if d1 > 0:
        out += (x - t[i]) / d1 * _bspline_one(x, t, i, k - 1, right)
    d2 = t[i + k + 1] - t[i + 1]
    if d2 > 0:
        out += (t[i + k + 1] - x) / d2 * _bspline_one(x, t, i + 1, k - 1,
                                                      right)
    return out


def _diff_penalty(m, order=2):
    D = np.eye(m)
    for _ in range(int(order)):
        D = np.diff(D, axis=0)
    return D.T @ D


def _smoother(B, P, lam):
    """Penalised-spline hat matrix S = B (B'B + lam P)^-1 B'."""
    A = B.T @ B + lam * P
    try:
        M = np.linalg.solve(A, B.T)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(A) @ B.T
    return B @ M


def face_smooth(Y, argvals=None, n_basis=12, degree=3, lambdas=None,
                pve=0.99, penalty_order=2):
    r"""Smooth a functional covariance and extract its components.

    Given curves observed on a common grid, the raw covariance
    :math:`\hat C = n^{-1} \tilde Y^\top \tilde Y` is smoothed by a
    *sandwich*:

    .. math:: \hat C_{\text{smooth}} = S \hat C S^\top,

    with :math:`S` a penalised-spline smoother. Smoothing both margins
    with the same operator is what makes this fast -- the two-dimensional
    problem never has to be assembled -- and it keeps the result
    symmetric by construction.

    **The diagonal must be excluded before smoothing, and this is the
    whole point of the method.** With white measurement error
    :math:`\epsilon` of variance :math:`\sigma^2`, the raw covariance
    satisfies

    .. math::
        E[\hat C(s,t)] = C(s,t) + \sigma^2 \mathbb{1}\{s = t\},

    so the diagonal sits a constant :math:`\sigma^2` above the surface
    everywhere while the off-diagonal is unbiased. Smoothing straight
    through it drags that ridge outwards into the neighbouring
    entries, inflating the leading eigenvalue and biasing every
    eigenfunction. Dropping the diagonal, smoothing, and then reading
    :math:`\sigma^2` off the gap between the raw and smoothed diagonals
    estimates the noise instead of absorbing it.
    ``noise_variance_bias_if_kept`` reports what including it would
    have cost on this data.

    The smoothed covariance is then eigendecomposed for functional
    principal components. Negative eigenvalues are truncated to zero,
    because a covariance operator cannot have any and their appearance
    is a symptom of smoothing, not a feature of the data --
    ``negative_eigenvalue_mass`` says how much was removed.

    Parameters
    ----------
    Y : array-like, shape (n_curves, n_points)
        Observed curves on a common grid. NaN is allowed.
    argvals : array-like, optional
        Grid points. Defaults to an equally spaced grid on [0, 1].
    n_basis, degree, penalty_order : int
        Spline basis size, degree, and difference-penalty order.
    lambdas : sequence of float, optional
        Smoothing parameters to search. The one minimising generalised
        cross-validation on the off-diagonal entries is used.
    pve : float
        Proportion of variance explained, for choosing the number of
        components to report.

    Returns
    -------
    RichResult
        ``covariance``, ``raw_covariance``, ``eigenvalues``,
        ``eigenfunctions``, ``noise_variance``, ``npc``, ``lambda``,
        ``mean_function``, ``scores``, ``pve_cumulative``.

    References
    ----------
    Xiao L, Zipunnikov V, Ruppert D, Crainiceanu C (2016)
    *Stat Comput* 26:409-421. Xiao L, Li Y, Ruppert D (2013)
    *JRSS B* 75(3):577-599. Yao F, Mueller HG, Wang JL (2005)
    *JASA* 100(470):577-590.
    """
    M = np.atleast_2d(np.asarray(Y, dtype=float))
    if M.ndim != 2:
        raise ValueError(f"Y must be 2-D; got shape {M.shape}.")
    n, p = M.shape
    if n < 2:
        raise ValueError(f"need at least two curves; got {n}.")
    if p < 4:
        raise ValueError(f"need at least four grid points; got {p}.")
    t = (np.linspace(0.0, 1.0, p) if argvals is None
         else np.asarray(argvals, dtype=float).ravel())
    if t.size != p:
        raise ValueError(
            f"argvals has length {t.size} but Y has {p} columns."
        )
    if not 0 < pve <= 1:
        raise ValueError(f"pve must lie in (0, 1]; got {pve}.")

    obs = np.isfinite(M)
    cnt = obs.sum(axis=0)
    if np.any(cnt < 2):
        raise ValueError(
            "every grid point needs at least two observed curves; points "
            f"{np.flatnonzero(cnt < 2).tolist()} do not."
        )
    mu = np.nansum(np.where(obs, M, 0.0), axis=0) / cnt
    Z = np.where(obs, M - mu, 0.0)

    # raw covariance with pairwise counts, so gaps do not shrink entries
    pair = obs.astype(float).T @ obs.astype(float)
    raw = (Z.T @ Z) / np.maximum(pair, 1.0)

    B = bspline_basis(t, n_basis=n_basis, degree=degree)
    P = _diff_penalty(B.shape[1], penalty_order)
    off = ~np.eye(p, dtype=bool)

    if lambdas is None:
        lambdas = np.logspace(-6, 4, 21)
    lambdas = np.asarray(lambdas, dtype=float).ravel()
    best = (np.inf, lambdas[0], None)
    for lam in lambdas:
        S = _smoother(B, P, float(lam))
        fit = S @ raw @ S.T
        resid = (raw - fit)[off]
        tr = float(np.trace(S))
        denom = (1.0 - tr / p) ** 2
        gcv = float(resid @ resid) / max(denom, 1e-12)
        if gcv < best[0]:
            best = (gcv, float(lam), S)
    gcv, lam, S = best

    # Smooth WITHOUT the diagonal, which carries the measurement error.
    # The diagonal is treated as missing and imputed from the smoothed
    # surface itself, iterating a few times. Filling it with the mean of
    # the whole row instead -- which is what a first pass at this
    # invites -- is badly wrong: the row mean of C(s, t) over all s is
    # nowhere near C(t, t), and on a two-component Karhunen-Loeve design
    # it inflated the estimated noise variance from 0.09 to 0.33 and
    # left 13 % of the eigenvalue mass negative.
    idx = np.arange(p)
    filled = raw.copy()
    # start from the average of the two adjacent off-diagonal entries,
    # which is local and therefore already close for a smooth surface
    nb = np.empty(p)
    for i in range(p):
        j = [k for k in (i - 1, i + 1) if 0 <= k < p]
        nb[i] = float(np.mean(raw[i, j]))
    filled[idx, idx] = nb
    C = S @ filled @ S.T
    C = 0.5 * (C + C.T)
    for _ in range(10):
        filled[idx, idx] = np.diag(C)
        C_new = S @ filled @ S.T
        C_new = 0.5 * (C_new + C_new.T)
        if np.max(np.abs(np.diag(C_new) - np.diag(C))) < 1e-12:
            C = C_new
            break
        C = C_new

    # estimate the noise variance away from the boundary, where the
    # smoother is least reliable
    keep_lo, keep_hi = int(0.1 * p), int(0.9 * p)
    if keep_hi <= keep_lo:
        keep_lo, keep_hi = 0, p
    gap = np.diag(raw) - np.diag(C)
    sigma2 = float(np.mean(np.maximum(gap[keep_lo:keep_hi], 0.0)))

    # what smoothing through the diagonal would have done instead
    C_kept = S @ raw @ S.T
    C_kept = 0.5 * (C_kept + C_kept.T)
    bias_if_kept = float(np.mean(np.diag(C_kept) - np.diag(C)))

    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    neg_mass = float(np.sum(np.abs(vals[vals < 0])))
    vals = np.maximum(vals, 0.0)
    total = float(np.sum(vals))
    cum = np.cumsum(vals) / total if total > 0 else np.zeros_like(vals)
    npc = int(np.searchsorted(cum, pve) + 1) if total > 0 else 0
    npc = max(1, min(npc, p))

    # scale eigenvectors to eigenfunctions on the grid
    dt = float(np.mean(np.diff(t))) if p > 1 else 1.0
    phi = vecs / math.sqrt(max(dt, 1e-300))
    scores = np.where(obs, M - mu, 0.0) @ vecs[:, :npc]

    out = RichResult(
        title="FACE covariance smoothing",
        summary_lines=[
            ("Curves", n),
            ("Grid points", p),
            ("Smoothing parameter", lam),
            ("Noise variance", sigma2),
            ("Components for %.0f%% variance" % (pve * 100), npc),
        ],
        payload={
            "covariance": C,
            "raw_covariance": raw,
            "covariance_diagonal_kept": C_kept,
            "estimate": C,
            "eigenvalues": vals,
            "eigenfunctions": phi,
            "scores": scores,
            "mean_function": mu,
            "noise_variance": sigma2,
            "noise_variance_bias_if_kept": bias_if_kept,
            "negative_eigenvalue_mass": neg_mass,
            "npc": npc,
            "pve_cumulative": cum,
            "total_variance": total,
            "lambda": lam,
            "gcv": gcv,
            "smoother_trace": float(np.trace(S)),
            "n_basis": int(B.shape[1]),
            "argvals": t,
            "n": n,
            "method": _METHOD,
        },
        interpretation=(
            f"{npc} component(s) carry {cum[npc - 1] * 100:.1f}% of the "
            f"smoothed variance, with measurement-error variance estimated "
            f"at {sigma2:.4g}."
            if total > 0 else "The smoothed covariance has no positive mass."
        ),
    )
    if neg_mass > 1e-8 * max(total, 1e-300):
        out.warnings.append(
            f"The smoothed covariance had negative eigenvalues carrying "
            f"{neg_mass:.4g} of mass, truncated to zero. A covariance "
            "operator cannot have any; their presence means the smoother is "
            "not projecting onto a valid covariance."
        )
    if sigma2 <= 0:
        out.warnings.append(
            "The smoothed diagonal sits at or above the raw diagonal, so no "
            "measurement error is detectable. Either the curves are noise "
            "free or the smoother is under-smoothing and tracking the noise."
        )
    if lam == lambdas[0] or lam == lambdas[-1]:
        out.warnings.append(
            f"The chosen smoothing parameter {lam:g} sits at the edge of the "
            "search grid, so the minimum of the criterion may lie outside it."
        )
    return out


def cheatsheet():
    return (
        "facea: FACE sandwich smoother for a functional covariance, holding "
        "the diagonal out so measurement error is estimated rather than "
        "smoothed into the surface"
    )
