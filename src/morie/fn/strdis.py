"""DeltaCon structural distance between two graphs (Koutra et al.)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["strdis", "structural_distance"]


def strdis(G1, G2, eps=None):
    """
    DeltaCon-0 graph similarity via fast belief propagation affinities.

    For each graph the node-affinity matrix is the FaBP linearization

        S = [I + eps^2 D - eps A]^{-1}                     (eq. 2.2)

    with A the adjacency matrix, D the diagonal degree matrix and
    eps = 1/(1 + max_i d_ii) (Table 1 of the paper; here the max runs
    over BOTH graphs so a single constant is shared, keeping the
    measure symmetric). The distance is the root Euclidean (Matusita)
    distance

        d = sqrt( sum_ij ( sqrt(s1_ij) - sqrt(s2_ij) )^2 )   (eq. 3.3)

    and the similarity is sim = 1/(1 + d) (Algorithm 1, DeltaCon-0).

    Sources
    -------
    Koutra, D., Vogelstein, J. T. & Faloutsos, C. (2013). DeltaCon: A
    principled massive-graph similarity function. *SIAM SDM 2013*,
    arXiv:1304.4657, eqs. (2.2), (3.3), Table 1, Algorithm 1
    (fetched-wave3/koutra-2013-deltacon.pdf).

    Parameters
    ----------
    G1, G2 : array-like, (n, n)
        Adjacency matrices on the same node set.
    eps : float, optional
        Override the influence constant.

    Returns
    -------
    RichResult
        Keys: distance, similarity (= estimate), eps.
    """
    A1 = np.atleast_2d(np.asarray(G1, dtype=float))
    A2 = np.atleast_2d(np.asarray(G2, dtype=float))
    n = A1.shape[0]
    if A1.shape != (n, n) or A2.shape != (n, n):
        raise ValueError("G1 and G2 must be square matrices of equal size")
    d1 = A1.sum(axis=1)
    d2 = A2.sum(axis=1)
    if eps is None:
        dmax = max(float(np.max(d1)), float(np.max(d2)))
        eps = 1.0 / (1.0 + dmax)
    eps = float(eps)

    def _affinity(A, d):
        M = np.eye(n) + eps**2 * np.diag(d) - eps * A
        return np.linalg.inv(M)

    S1 = _affinity(A1, d1)
    S2 = _affinity(A2, d2)
    dist = float(np.sqrt(np.sum((np.sqrt(np.abs(S1)) - np.sqrt(np.abs(S2))) ** 2)))
    sim = 1.0 / (1.0 + dist)
    return RichResult(payload={
        "distance": dist, "similarity": sim, "estimate": sim,
        "eps": eps, "n": int(n),
        "method": "DeltaCon-0 (FaBP affinities, RootED distance)",
    })


# long descriptive alias (stub-era name)
structural_distance = strdis


def cheatsheet():
    return "strdis: DeltaCon-0 similarity, S = inv(I + eps^2 D - eps A), RootED"
