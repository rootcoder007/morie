"""Graphical lasso (ESL Alg. 17.2)."""

from ._richresult import RichResult

__all__ = ["esl_graphical_lasso"]


def esl_graphical_lasso(S, lambda_, max_iter=1000, tol=1e-12, max_inner=10000):
    r"""Maximise :math:`\log\det\Theta - \mathrm{trace}(S\Theta) - \lambda\|\Theta\|_1` (ESL eq 17.21).

    ESL Alg. 17.2 (Friedman, Hastie & Tibshirani 2008): W starts at
    :math:`S + \lambda I` and keeps that diagonal; each column solves the
    modified lasso :math:`W_{11}\beta - s_{12} + \lambda\,\mathrm{Sign}(\beta) = 0` by
    the coordinate descent of eq 17.26,
    :math:`\hat\beta_j \leftarrow S(s_{12j} - \sum_{k\neq j}V_{kj}\hat\beta_k, \lambda)/V_{jj}`,
    then :math:`w_{12} = W_{11}\hat\beta`; finally
    :math:`\hat\theta_{22} = 1/(w_{22} - w_{12}^T\hat\beta)`, :math:`\hat\theta_{12} = -\hat\beta\hat\theta_{22}`.
    The penalty includes the diagonal, as ``glasso``'s default.

    Parameters
    ----------
    S : p x p nested sequence
    lambda_ : float
        Penalty, >= 0.
    max_iter, tol, max_inner
        Outer (largest change in W) and inner coordinate-descent controls.

    Returns
    -------
    RichResult
        ``Sigma`` (W), ``Theta``, ``iterations``, ``converged``.

    References
    ----------
    Friedman, J., Hastie, T. & Tibshirani, R. (2008). Sparse inverse
    covariance estimation with the graphical lasso. Biostatistics 9, 432-441.
    """
    Sm = [[float(v) for v in r] for r in S]
    p = len(Sm)
    lam = float(lambda_)
    if any(len(r) != p for r in Sm) or lam < 0:
        raise ValueError("S must be square and lambda >= 0")
    W = [[Sm[i][j] + (lam if i == j else 0.0) for j in range(p)] for i in range(p)]
    B = [[0.0] * p for _ in range(p)]
    it, converged = 0, False

    def soft(x, t):
        return (abs(x) - t) * (1 if x > 0 else -1) if abs(x) > t else 0.0

    for _ in range(max_iter):
        it += 1
        delta = 0.0
        for j in range(p):
            idx = [k for k in range(p) if k != j]
            b = [B[j][k] for k in idx]
            for _ in range(max_inner):
                dmax = 0.0
                for a, ka in enumerate(idx):
                    r = Sm[ka][j] - sum(W[ka][kb] * b[c] for c, kb in enumerate(idx) if c != a)
                    new = soft(r, lam) / W[ka][ka]
                    dmax = max(dmax, abs(new - b[a]))
                    b[a] = new
                if dmax < tol:
                    break
            for a, ka in enumerate(idx):
                B[j][ka] = b[a]
            for ka in idx:
                new = sum(W[ka][kb] * B[j][kb] for kb in idx)
                delta = max(delta, abs(new - W[ka][j]))
                W[ka][j] = W[j][ka] = new
        if delta < tol:
            converged = True
            break
    Th = [[0.0] * p for _ in range(p)]
    for j in range(p):
        t22 = 1 / (W[j][j] - sum(W[j][k] * B[j][k] for k in range(p) if k != j))
        Th[j][j] = t22
        for k in range(p):
            if k != j:
                Th[k][j] = -B[j][k] * t22
    Th = [[0.5 * (Th[i][j] + Th[j][i]) for j in range(p)] for i in range(p)]
    return RichResult(
        title="Graphical lasso",
        summary_lines=[("iterations", it)],
        payload={"Sigma": W, "Theta": Th, "iterations": it, "converged": converged},
    )


def cheatsheet():
    return "eslglso: W = S + lambda I; per-column lasso W11 b - s12 + lambda sign(b) = 0 (17.26)"
