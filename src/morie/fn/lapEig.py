"""Laplacian eigenmap embedding (Belkin-Niyogi)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["lapEig", "laplacian_eigenmaps"]


def _jacobi_eigh(S, max_sweeps=100, tol=1e-13):
    """Classical Jacobi eigen-decomposition of a symmetric matrix.

    Deterministic cyclic sweeps over the upper triangle (row-major
    order); rotation skipped when |S_pq| <= tol * sqrt(|S_pp S_qq|).
    Identical arithmetic is mirrored in the R arm so both arms return
    bit-comparable results (LAPACK eigenvector conventions are NOT
    relied upon).
    """
    n = S.shape[0]
    a = [[float(S[i, j]) for j in range(n)] for i in range(n)]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                off += a[p][q] * a[p][q]
        if off <= tol * tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp = V[k][p]
                    vkq = V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    vals = [a[i][i] for i in range(n)]
    return vals, V


def lapEig(A, k=2):
    """
    Laplacian eigenmap: embed graph nodes by the low eigenvectors of the
    generalized Laplacian problem.

    With W the (symmetric, non-negative) weight matrix, D = diag(W 1)
    and L = D - W, the embedding solves

        L f = lambda D f                                  (paper Sec. 2, step 3)

    and uses the eigenvectors f_1, ..., f_k for the eigenvalues
    0 = lambda_0 <= lambda_1 <= ... (the trivial constant f_0 is
    dropped). Computed here through the equivalent symmetric problem
    L_sym = D^{-1/2} L D^{-1/2}, whose eigenvectors v give f = D^{-1/2} v,
    solved with a deterministic cyclic Jacobi rotation (mirrored
    exactly in the R arm). Sign convention: each eigenvector is scaled
    so its largest-magnitude entry is positive.

    Sources
    -------
    Belkin, M. & Niyogi, P. (2003). Laplacian eigenmaps for
    dimensionality reduction and data representation. *Neural
    Computation*, 15(6), 1373-1396, Sec. 2 (algorithm step 3:
    "Eigenmaps: ... solve the generalized eigenvector problem
    L f = lambda D f") and Sec. 3 (justification via L_sym)
    (fetched-wave3/belkin-niyogi-2003-laplacian-eigenmaps.pdf).

    Parameters
    ----------
    A : array-like, (n, n)
        Symmetric non-negative weight/adjacency matrix; every node
        needs positive degree.
    k : int
        Embedding dimension (number of non-trivial eigenvectors).

    Returns
    -------
    RichResult
        Keys: embedding (n x k, columns = f_1..f_k), eigenvalues
        (lambda_1..lambda_k), all_eigenvalues (ascending).
    """
    W = np.atleast_2d(np.asarray(A, dtype=float))
    n = W.shape[0]
    if W.shape[1] != n:
        raise ValueError("A must be square")
    k = int(k)
    if not (1 <= k < n):
        raise ValueError("k must satisfy 1 <= k < n")
    d = np.sum(W, axis=1)
    if np.any(d <= 0):
        raise ValueError("every node must have positive degree")
    dis = 1.0 / np.sqrt(d)
    Lsym = np.eye(n) - (dis[:, None] * W) * dis[None, :]
    # symmetrize against roundoff before Jacobi
    Lsym = (Lsym + Lsym.T) / 2.0
    vals, V = _jacobi_eigh(Lsym)
    order = sorted(range(n), key=lambda i: vals[i])
    lam = [float(vals[i]) for i in order]
    emb = np.zeros((n, k))
    out_vals = []
    for c in range(k):
        idx = order[c + 1]  # drop the trivial lambda_0 = 0
        f = [dis[r] * V[r][idx] for r in range(n)]
        # sign fix: largest-|.| entry positive; ties by first index
        big = 0
        for r in range(1, n):
            if abs(f[r]) > abs(f[big]) + 1e-15:
                big = r
        sgn = 1.0 if f[big] >= 0 else -1.0
        for r in range(n):
            emb[r, c] = sgn * f[r]
        out_vals.append(lam[c + 1])
    return RichResult(payload={
        "embedding": emb, "eigenvalues": out_vals,
        "all_eigenvalues": lam, "k": k, "n": int(n),
        "method": "Laplacian eigenmap, generalized L f = lambda D f via Jacobi",
    })


# long descriptive alias (stub-era name)
laplacian_eigenmaps = lapEig


def cheatsheet():
    return "lapEig: Belkin-Niyogi Laplacian eigenmap, L f = lambda D f"
