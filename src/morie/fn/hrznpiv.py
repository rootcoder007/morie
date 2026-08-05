# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric IV model: Y=g(X)+U, E[U|W]=0."""

from . import _array_core as np
from . import _hrz3 as H

from ._richresult import RichResult

__all__ = ["horowitz_npiv_model"]


def horowitz_npiv_model(x, y, w, bandwidth=None, grid=25, tol=1e-5):
    r"""Nonparametric instrumental-variables model as a Fredholm
    equation of the first kind.

    Horowitz (2009), Section 5.3, pages 156-159.  For
    :math:`Y = g(X) + U` with :math:`E(U|W = w) = 0` and the support of
    :math:`(X, W)` mapped into :math:`[0,1]^2`,

    .. math:: E(Y|W = w)f_W(w) = \int_0^1 f_{XW}(x,w)g(x)\,dx  \quad (5.40)

    Multiplying by :math:`f_{XW}(z,w)` and integrating over :math:`w`
    symmetrises the problem into

    .. math:: r(z) = \int_0^1 \tau(x,z)g(x)\,dx,               \quad (5.41)

    .. math:: r(z) = \int_0^1 E(Y|W=w)f_{XW}(z,w)f_W(w)\,dw,   \quad (5.42)

    .. math:: \tau(x,z) = \int_0^1 f_{XW}(x,w)f_{XW}(z,w)\,dw, \quad (5.43)

    that is :math:`r = Tg` (5.44) with :math:`T` self-adjoint and
    positive semi-definite by (5.45).

    Estimation here is by SPECTRAL TRUNCATION, which is the
    regularisation the section itself motivates: :math:`T` has
    eigenvalues :math:`\lambda_j \downarrow 0`, so
    :math:`T^{-1}h = \sum_j \langle h,\phi_j\rangle \phi_j/\lambda_j`
    exists only formally, and terms with tiny :math:`\lambda_j`
    amplify estimation noise without bound.  Truncating at
    :math:`\lambda_j > \texttt{tol}\cdot\lambda_1` is what makes the
    inverse usable.  ``n_terms`` reports how many terms survived and
    ``eigenvalues`` exposes the decay, because that decay IS the
    ill-posedness rather than an incidental diagnostic.

    Singularity of :math:`T` -- a zero eigenvalue -- is exactly the
    failure of identification (pp. 158-159): if :math:`Th = 0` for
    some :math:`h \neq 0` then :math:`g` and :math:`g + h` satisfy
    (5.40) equally well.  ``identified`` reports whether the smallest
    eigenvalue clears the tolerance.

    Parameters
    ----------
    x : array-like, shape (n,)
        Endogenous regressor.
    y : array-like, shape (n,)
        Response.
    w : array-like, shape (n,)
        Instrument.
    bandwidth : float, optional
        Kernel bandwidth on the [0, 1] scale; default
        ``1.06 n^(-1/6)/sqrt(12)``.
    grid : int, default 25
        Number of quadrature points on [0, 1].
    tol : float, default 1e-5
        Relative eigenvalue cutoff for the truncated inverse.  This
        bounds the amplification factor :math:`\lambda_1/\lambda_j`
        of the retained directions at ``1/tol``.  The default is not
        cosmetic: at ``tol = 1e-8`` this estimator retains a seventh
        direction that amplifies by :math:`6\times10^{7}`, and two
        correct implementations of the SAME formula then disagree in
        the seventh decimal of ``g_hat`` purely through rounding.
        That is the ill-posedness of Section 5.3 showing up as
        arithmetic, and the cutoff is what controls it.

    Returns
    -------
    RichResult
        keys: ``g_hat``, ``grid_points``, ``r_hat``, ``eigenvalues``,
        ``fW``, ``raw_mass``, ``trace_T``, ``n_terms``, ``identified``, ``bandwidth``, ``n``,
        ``m``, ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 5.3, eqs. (5.40)-(5.46),
    pp. 156-159.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = int(x.size)
    if y.size != n or w.size != n:
        raise ValueError(
            f"x, y, w must have the same length; got {n}, {y.size}, {w.size}.")
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    h = float(bandwidth) if bandwidth is not None else H.bw01(n)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    tol = float(tol)
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")

    u = H.u01(x)
    v = H.u01(w)
    z, wq = H.grid_w(grid)
    m = int(z.size)

    KW = H.kmat(z, v, h)

    # f_hat_XW on the grid, mass-corrected at the [0,1]^2 boundary.
    fxw, raw_mass = H.fxw_grid(u, v, z, wq, h)

    # f_W(w) = int f_XW(x, w) dx.
    fW = [0.0] * m
    for l in range(m):
        s = 0.0
        for k in range(m):
            s += float(wq[k]) * fxw[k][l]
        fW[l] = s

    # E(Y|W = w) by Nadaraya-Watson on the [0,1] scale.
    mW = [0.0] * m
    for l in range(m):
        num = den = 0.0
        for i in range(n):
            num += float(KW[l][i]) * float(y[i])
            den += float(KW[l][i])
        mW[l] = num / den if den > 1e-300 else 0.0

    # r(z), eq. (5.42), and tau(x, z), eq. (5.43).
    r_hat = [0.0] * m
    for l in range(m):
        s = 0.0
        for q in range(m):
            s += float(wq[q]) * mW[q] * fxw[l][q] * fW[q]
        r_hat[l] = s
    tau = [[0.0] * m for _ in range(m)]
    for k in range(m):
        for l in range(m):
            s = 0.0
            for q in range(m):
                s += float(wq[q]) * fxw[k][q] * fxw[l][q]
            tau[k][l] = s

    # Symmetrise in the quadrature inner product: S = D^(1/2) tau D^(1/2),
    # so S's eigenpairs give T's eigenpairs with phi = D^(-1/2) s.
    rt = [float(wq[k]) ** 0.5 for k in range(m)]
    S = [[rt[k] * tau[k][l] * rt[l] for l in range(m)] for k in range(m)]
    lam, vecs = np.linalg.eigh(np.asarray(S, dtype=float))
    order = sorted(range(m), key=lambda j: -float(lam[j]))
    lam_s = [float(lam[j]) for j in order]

    trace_T = 0.0
    for k in range(m):
        trace_T += float(wq[k]) * tau[k][k]

    cut = tol * lam_s[0] if lam_s[0] > 0 else 0.0
    g_hat = [0.0] * m
    n_terms = 0
    for j in order:
        lj = float(lam[j])
        if lj <= cut:
            continue
        n_terms += 1
        # phi_j = D^(-1/2) s_j; <r, phi_j>_wq = sum_k wq_k r_k phi_j[k].
        ip = 0.0
        for k in range(m):
            ip += float(wq[k]) * r_hat[k] * (float(vecs[k][j]) / rt[k])
        c = ip / lj
        for k in range(m):
            g_hat[k] += c * (float(vecs[k][j]) / rt[k])

    return RichResult(payload={
        "g_hat": g_hat,
        "grid_points": [float(t) for t in z],
        "r_hat": r_hat,
        "eigenvalues": lam_s,
        "trace_T": trace_T,
        "n_terms": n_terms,
        "identified": bool(lam_s[m - 1] > cut),
        "fW": fW,
        "raw_mass": raw_mass,
        "bandwidth": h,
        "n": n,
        "m": m,
        "method": "Horowitz (2009) eqs. (5.41)-(5.44), spectral truncation of T",
    })


def cheatsheet():
    return "hrznpiv: r = Tg (5.44); the eigenvalue decay of T IS the ill-posedness"
