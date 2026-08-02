# morie.fn -- function file (rootcoder007/morie)
"""Wordfish: Poisson scaling of word-frequency matrices."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["em_irt_text"]


def em_irt_text(word_freq_matrix, max_iter=200, tol=1e-6, polarity=(0, 1)):
    r"""Slapin-Proksch wordfish by alternating Poisson regressions.

    .. math:: y_{ik} \sim \mathrm{Poisson}\big(
              \exp(\alpha_i + \psi_k + \beta_k \theta_i)\big),

    with document positions :math:`\theta_i`, document fixed effects
    :math:`\alpha_i`, word fixed effects :math:`\psi_k` and word
    discriminations :math:`\beta_k`. The likelihood separates given
    the other block, so estimation alternates Newton steps for the
    word parameters and the document parameters. Identification:
    :math:`\theta` standardised each sweep, :math:`\alpha_1 = 0`, and
    the polarity pair orders ``theta[polarity[0]] <
    theta[polarity[1]]``.

    Parameters
    ----------
    word_freq_matrix : array-like of nonnegative ints, shape (n, k)
        Document x word count matrix.
    max_iter : int, default 200
    tol : float, default 1e-6
        Stop when theta moves less than tol.
    polarity : (int, int), default (0, 1)
        Document pair fixing the direction of the scale.

    Returns
    -------
    RichResult
        keys: ``theta`` (n,), ``alpha`` (n,), ``psi`` (k,), ``beta``
        (k,), ``n_iter``, ``converged``, ``n_docs``, ``n_words``,
        ``method``.

    References
    ----------
    Slapin, J. B. & Proksch, S.-O. (2008). A scaling model for
    estimating time-series party positions from texts. *AJPS*, 52(3),
    705-722.
    """
    Y = np.asarray(word_freq_matrix, dtype=float)
    if Y.ndim != 2:
        raise ValueError("word_freq_matrix must be 2-D (documents x words).")
    if np.any(Y < 0) or np.any(Y != np.floor(Y)):
        raise ValueError("word_freq_matrix must hold nonnegative counts.")
    n, k = Y.shape
    if n < 3 or k < 3:
        raise ValueError("need at least 3 documents and 3 words.")
    p0, p1 = (int(p) for p in polarity)
    if p0 == p1 or not (0 <= p0 < n and 0 <= p1 < n):
        raise ValueError("polarity must be two distinct document indices.")

    # starting values from row/column structure (Slapin-Proksch appendix)
    alpha = np.log(np.maximum(Y.sum(axis=1), 1.0))
    alpha = alpha - alpha[0]
    psi = np.log(np.maximum(Y.mean(axis=0), 1e-8))
    # theta from the first factor of the residual log counts
    L = np.log(Y + 0.5) - alpha[:, None] - psi[None, :]
    u, s, vt = np.linalg.svd(L, full_matrices=False)
    theta = u[:, 0]
    theta = (theta - theta.mean()) / max(theta.std(), 1e-8)
    beta = vt[0] * s[0] / max(np.linalg.norm(u[:, 0]), 1e-8)

    def newton(offset, xcol, y, coef, max_steps=25):
        """One-column Poisson regression y ~ offset + coef * xcol (updates coef pieces)."""
        b = coef.copy()
        for _ in range(max_steps):
            eta = np.clip(offset + xcol @ b, -30, 30)
            mu = np.exp(eta)
            grad = xcol.T @ (y - mu)
            H = (xcol * mu[:, None]).T @ xcol + 1e-8 * np.eye(xcol.shape[1])
            step = np.linalg.solve(H, grad)
            b = b + step
            if np.max(np.abs(step)) < 1e-10:
                break
        return b

    converged = False
    for it in range(int(max_iter)):
        theta_old = theta.copy()
        # word block: (psi_k, beta_k) given documents
        Xw = np.column_stack([np.ones(n), theta])
        for kk in range(k):
            psi[kk], beta[kk] = newton(alpha, Xw, Y[:, kk], np.array([psi[kk], beta[kk]]))
        # document block: (alpha_i, theta_i) given words; alpha_1 fixed at 0
        Xd = np.column_stack([np.ones(k), beta])
        for i in range(n):
            coef = newton(psi, Xd, Y[i], np.array([alpha[i], theta[i]]))
            if i == 0:
                theta[i] = coef[1]
                alpha[i] = 0.0
            else:
                alpha[i], theta[i] = coef
        # identify theta
        mu, sd = theta.mean(), max(theta.std(), 1e-8)
        theta = (theta - mu) / sd
        psi = psi + beta * mu
        beta = beta * sd
        if theta[p0] > theta[p1]:
            theta = -theta
            beta = -beta
        if np.max(np.abs(theta - theta_old)) < tol:
            converged = True
            break

    return RichResult(
        payload={
            "theta": theta,
            "alpha": alpha,
            "psi": psi,
            "beta": beta,
            "n_iter": it + 1,
            "converged": converged,
            "n_docs": int(n),
            "n_words": int(k),
            "method": "Wordfish Poisson scaling (alternating Newton blocks)",
        }
    )


def cheatsheet():
    return "emtxt: y ~ Poisson(exp(alpha_i + psi_k + beta_k theta_i)) (Slapin-Proksch 2008)"
