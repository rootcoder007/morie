# morie.fn -- function file (rootcoder007/morie)
"""Spectral clustering for the stochastic block model."""

import numpy as np

from ._richresult import RichResult

__all__ = ["spectral_sbm"]


def spectral_sbm(adjacency, k=2, regularized=True, n_iter=100, seed=0):
    r"""Recover blocks from the leading eigenvectors of the adjacency.

    The rows of the top-:math:`K` eigenvector matrix cluster by block,
    because the expected adjacency of an SBM has rank :math:`K` and its
    eigenvectors are constant within blocks. Lei and Rinaldo prove that
    spectral clustering on the adjacency matrix is consistent provided
    the average degree grows faster than :math:`\log n`.

    That degree condition is the practical constraint and it is
    checked. Below it the graph is too sparse for the leading
    eigenvectors to separate the blocks -- the spectrum is dominated by
    high-degree vertices rather than by community structure, and the
    method fails silently by returning a partition that mostly reflects
    degree.

    REGULARISATION is the standard remedy: adding :math:`\tau/n` to
    every entry (with :math:`\tau` the mean degree) shrinks the
    influence of low-degree vertices and extends consistency into the
    sparse regime. It is on by default and ``regularized`` records it.

    ``eigengap`` between the :math:`K`-th and :math:`(K+1)`-th
    eigenvalue is the evidence that :math:`K` was the right number of
    blocks; a small gap means the choice was not supported by the
    spectrum, whatever the clustering looks like.

    Parameters
    ----------
    adjacency : array-like, shape (n, n)
        Symmetric, zero diagonal.
    k : int
        Number of blocks.
    regularized : bool
    n_iter : int
        k-means iterations.
    seed : int

    Returns
    -------
    RichResult
        ``labels``, ``eigenvalues``, ``eigengap``, ``mean_degree``,
        ``degree_condition``, ``modularity``, ``block_matrix``.

    References
    ----------
    Lei and Rinaldo (2015), *Annals of Statistics* 43:215-237.
    Rohe, Chatterjee and Yu (2011), *Annals of Statistics*
    39:1878-1915.
    Amini et al. (2013) for regularisation.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.zeros((6, 6))
    >>> A[:3, :3] = 1 - np.eye(3)
    >>> A[3:, 3:] = 1 - np.eye(3)
    >>> len(set(spectral_sbm(A, k=2)["labels"].tolist()))
    2
    """
    A = np.atleast_2d(np.asarray(adjacency, dtype=float))
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("adjacency must be square, got %s." % (A.shape,))
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("adjacency must be symmetric.")
    if np.any(np.abs(np.diag(A)) > 1e-12):
        raise ValueError("adjacency must have a zero diagonal.")
    K = int(k)
    if K < 1 or K > n:
        raise ValueError("k must lie between 1 and n, got %d." % K)

    deg = A.sum(axis=1)
    mean_deg = float(deg.mean())
    thresh = float(np.log(n)) if n > 1 else 1.0
    ok = mean_deg > thresh

    M = A.copy()
    if regularized:
        M = M + mean_deg / n

    w, V = np.linalg.eigh(M)
    order = np.argsort(np.abs(w))[::-1]
    w, V = w[order], V[:, order]
    U = V[:, :K]
    nrm = np.linalg.norm(U, axis=1, keepdims=True)
    Un = U / np.maximum(nrm, 1e-12)

    rng = np.random.default_rng(int(seed))
    centres = Un[rng.choice(n, size=K, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(int(n_iter)):
        d2 = ((Un[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(K):
            m = labels == c
            if m.any():
                centres[c] = Un[m].mean(axis=0)

    B = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            ma, mb = labels == a, labels == b
            if ma.any() and mb.any():
                blk = A[np.ix_(ma, mb)]
                cnt = ma.sum() * mb.sum() - (ma.sum() if a == b else 0)
                B[a, b] = blk.sum() / max(cnt, 1)

    tot = A.sum()
    q = 0.0
    if tot > 0:
        for c in range(K):
            m = labels == c
            q += A[np.ix_(m, m)].sum() / tot - (deg[m].sum() / tot) ** 2
    gap = float(abs(w[K - 1]) - abs(w[K])) if K < n else np.nan
    return RichResult(
        payload={
            "estimate": labels,
            "labels": labels,
            "eigenvalues": w[:min(K + 3, n)],
            "eigengap": gap,
            "eigengap_note": (
                "the gap between the K-th and (K+1)-th eigenvalue is the "
                "spectrum's evidence for K; a small gap means K was not "
                "supported however clean the clustering looks"
            ),
            "mean_degree": mean_deg,
            "log_n": thresh,
            "degree_condition": bool(ok),
            "degree_note": (
                None if ok else
                "mean degree %.2f is below log n = %.2f; the graph is too "
                "sparse for the leading eigenvectors to separate blocks, and "
                "the partition will largely reflect degree instead"
                % (mean_deg, thresh)
            ),
            "regularized": bool(regularized),
            "regularization_note": (
                "adding tau/n to every entry shrinks low-degree vertices' "
                "influence and extends consistency into the sparse regime"
            ),
            "block_matrix": B,
            "modularity": float(q),
            "block_sizes": np.bincount(labels, minlength=K),
            "k": K,
            "n": int(n),
            "method": "Spectral stochastic block model recovery",
        }
    )


def cheatsheet():
    return (
        "sgtsbms: spectral SBM clustering with the log-n degree condition "
        "and the eigengap that justifies K"
    )
