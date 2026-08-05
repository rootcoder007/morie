# morie.fn -- function file (rootcoder007/morie)
"""Matrix completion for causal panel data (MC-NNM)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["matrix_completion_scm", "matrixcompletionscm"]


def matrix_completion_scm(y, D, lam, max_iter=500, tol=1e-10):
    """Impute the untreated counterfactual panel by nuclear-norm completion.

    Athey et al. treat causal panel estimation as a matrix completion
    problem: the untreated potential outcomes form one matrix ``L``,
    of which the control cells are observed and the treated cells are
    missing.  Estimating the treatment effect is then filling in the
    holes.  Their estimator regularises with the nuclear norm, the
    convex relaxation of rank:

        min_L  (1/|O|) sum_{(i,t) in O} (Y_it - L_it)^2 + lam ||L||_*

    with ``O`` the set of untreated (observed) cells.  The solution is
    reached by the standard soft-impute iteration: fill the missing
    cells with the current estimate, take the SVD of the completed
    matrix, and soft-threshold its singular values by ``lam``,

        L <- U diag(max(s - lam, 0)) V',

    repeating until the Frobenius change falls below ``tol``.  Singular
    value THRESHOLDING, not truncation, is what makes this the exact
    proximal step of the nuclear norm: truncating to a fixed rank would
    solve a different, non-convex problem.

    The iteration is fully deterministic -- a zero start, a fixed
    schedule, a fixed tolerance -- so both language arms land on the
    same numbers rather than merely the same distribution.  The SVD's
    sign convention cannot separate them either: ``U diag(s) V'`` is
    invariant to flipping the sign of a matched column pair.

    ``lam = 0`` is a DEGENERATE limit, not the unregularised ideal.
    With no shrinkage the SVD step reconstructs its input exactly, so
    the iteration reaches ``L = P_O(Y)`` -- observed cells reproduced,
    missing cells left at their zero start -- after two passes and stops.
    The reported ATT is then simply the raw mean of ``Y`` over the
    treated cells, with no counterfactual having been imputed at all.
    It is retained because it is a sharp check that the projection step
    is doing what it claims, but it must not be read as an estimate.
    Any real use wants ``lam`` on the order of the panel's singular
    values.

    Parameters
    ----------
    y : array-like, shape (N, T)
        Observed outcome panel, units by periods.
    D : array-like, shape (N, T)
        Treatment indicator, 1 where the cell is treated (and its
        untreated outcome therefore missing).
    lam : float
        Nuclear-norm penalty, in the units of a singular value.
    max_iter : int, default 500
        Iteration cap.
    tol : float, default 1e-10
        Frobenius convergence tolerance on successive iterates.

    Returns
    -------
    RichResult
        ``estimate`` (the ATT over treated cells), ``att``, ``L``
        (completed matrix), ``tau`` (per-treated-cell effects),
        ``n_treated``, ``rank``, ``nuclear``, ``iterations``,
        ``converged``, ``N``, ``T``.

    References
    ----------
    Athey, S., Bayati, M., Doudchenko, N., Imbens, G. and Khosravi, K.
    (2021), "Matrix Completion Methods for Causal Panel Data Models",
    Journal of the American Statistical Association 116(536),
    1716-1730, doi:10.1080/01621459.2021.1891924, verified against
    Crossref (the earlier NBER working paper is w25132,
    doi:10.3386/w25132).  The article was not in the local corpus; the
    objective and the soft-impute step above are its standard published
    form, stated in full so they can be checked against it.
    """
    Y = np.asarray(y, dtype=float)
    W = np.asarray(D, dtype=float)
    if Y.ndim != 2:
        raise ValueError("y must be a 2-D panel, units by periods")
    if W.shape != Y.shape:
        raise ValueError("D must have the same shape as y")
    N, T = Y.shape
    lm = float(lam)
    if lm < 0.0:
        raise ValueError("lam must be non-negative")
    if not np.all((W == 0) | (W == 1)):
        raise ValueError("D must be binary 0/1")
    obs = W == 0
    n_obs = int(np.sum(obs))
    if n_obs == 0:
        raise ValueError("every cell is treated; there is nothing to learn from")
    n_treated = N * T - n_obs
    if n_treated == 0:
        raise ValueError("no cell is treated; there is no effect to estimate")

    L = np.zeros((N, T))
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        # Fill the treated (missing) cells with the current estimate and
        # keep the observed ones exactly: this is P_O(Y) + P_O^perp(L).
        Z = np.where(obs, Y, L)
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        s_th = np.maximum(s - lm, 0.0)
        Lnew = U @ np.diag(s_th) @ Vt
        diff = float(np.sqrt(np.sum((Lnew - L) ** 2)))
        L = Lnew
        if diff <= tol:
            converged = True
            break

    sv = np.linalg.svd(L, compute_uv=False)
    tau = []
    tot = 0.0
    for i in range(N):
        for t in range(T):
            if W[i, t] == 1:
                d = float(Y[i, t] - L[i, t])
                tau.append(d)
                tot += d
    att = tot / n_treated
    eps = 1e-10 * (float(sv[0]) if len(sv) else 1.0)
    return RichResult(payload={
        "estimate": att, "att": att, "L": [[float(L[i, t]) for t in range(T)]
                                           for i in range(N)],
        "tau": tau, "n_treated": n_treated, "n_observed": n_obs,
        "rank": int(np.sum(sv > eps)), "nuclear": float(np.sum(sv)),
        "iterations": it, "converged": 1.0 if converged else 0.0,
        "N": N, "T": T,
        "method": "Matrix completion for causal panel data (MC-NNM)"})


matrixcompletionscm = matrix_completion_scm


def cheatsheet():
    return "mscmcl: nuclear-norm matrix completion for causal panel data"
