# morie.fn -- function file (rootcoder007/morie)
"""DINO teacher centring and sharpening."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['dinocenter', 'dino_centering']


def dinocenter(g_t, center=None, m=0.9, tau_t=0.04):
    """DINO teacher centring and sharpening.

    Centring alone pushes the teacher toward the uniform distribution and sharpening alone pushes it toward a one-hot; either on its own collapses, and it is the two together that hold the output away from both degenerate fixed points. The centre used for the current batch is the incoming one and the update is returned separately, which is the order the paper's pseudocode uses -- applying the freshly updated centre to the same batch is a different and subtly self-referential algorithm. The paper writes the centre as C; the argument is named center because C is this batch's alias for its numeric helper module.


    Formula: P_t = softmax((g_t - C)/tau_t); C <- m C + (1-m) (1/B) sum_i g_t(x_i)

    Parameters
    ----------
    g_t : array-like, shape (B, K)
        Teacher logits, one row per view.
    center : array-like, optional
        Current centre C of length K; zeros if omitted.
    m : float
        Centre EMA rate.
    tau_t : float
        Teacher temperature.

    Returns
    -------
    RichResult
        ``p_t``, ``center``, ``center_old``, ``batch_mean``, ``B``, ``K``.

    References
    ----------
    Caron, Touvron, Misra, Jegou, Mairal, Bojanowski and Joulin (2021),
    Emerging Properties in Self-Supervised Vision Transformers,
    ICCV/arXiv:2104.14294.  Verified against the paper: equation (1)
    for the temperature softmax, equation (4) for the centre update,
    and Algorithm 1's pseudocode for the order of centre-then-sharpen.
    """
    G = C.mat(g_t)
    B = len(G); K = len(G[0])
    c = [0.0] * K if center is None else C.vec(center)
    if len(c) != K:
        raise ValueError("center must have length K")
    P = []
    for row in G:
        z = [(row[j] - c[j]) / float(tau_t) for j in range(K)]
        mx = max(z)
        e = [math.exp(v - mx) for v in z]
        s = sum(e)
        P.append([v / s for v in e])
    bm = [sum(G[i][j] for i in range(B)) / B for j in range(K)]
    newc = [float(m) * c[j] + (1.0 - float(m)) * bm[j] for j in range(K)]
    return RichResult(payload={
        "p_t": P, "center": newc, "center_old": c, "batch_mean": bm,
        "B": B, "K": K, "method": "DINO teacher centring and sharpening"})


dino_centering = dinocenter


def cheatsheet():
    return "dinoss: DINO teacher centring and sharpening."
