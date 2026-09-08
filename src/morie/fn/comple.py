# morie.fn -- function file (rootcoder007/morie)
"""ComplEx knowledge-graph embedding score."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["complex"]


def complex(triples, dim, re_e=None, im_e=None, re_r=None, im_r=None, seed=1):
    """Score knowledge-graph triples in a complex embedding space.

    A real bilinear model has to choose: make the relation matrix
    symmetric and lose antisymmetric relations, or make it full and pay
    ``d^2`` parameters.  ComplEx escapes the choice by moving to
    complex vectors and taking the real part of a Hermitian product.
    Conjugating the tail breaks the symmetry, so ``score(h, r, t)`` and
    ``score(t, r, h)`` differ while the parameter count stays linear in
    ``d``.

    Formula: ``score = Re(<e_h, w_r, conj(e_t)>)``, which expands to
    ``sum_k [Re h Re r Re t + Re h Im r Im t + Im h Re r Im t
    - Im h Im r Re t]``.

    Parameters
    ----------
    triples : array-like, shape (m, 3)
        Rows ``[head, relation, tail]`` as zero-based integer indices.
    dim : int
        Embedding dimension.
    re_e, im_e : array-like, optional
        Real and imaginary entity embeddings, shape (n_entities, dim).
    re_r, im_r : array-like, optional
        Real and imaginary relation embeddings, shape (n_relations, dim).
    seed : int, default 1
        Seed for the built-in generator used when embeddings are not
        supplied.  The stream is the shared Lehmer minstd, so both
        language arms produce identical defaults.

    Returns
    -------
    RichResult
        ``estimate`` (mean score), ``scores``, ``m``, ``dim``.

    References
    ----------
    Trouillon, T., Welbl, J., Riedel, S., Gaussier, E. & Bouchard, G.
    (2016).  Complex embeddings for simple link prediction.  ICML 33,
    2071-2080.  The score above is equation (11) of that paper.
    """
    T = [[int(v) for v in row] for row in C.mat(triples)]
    d = int(dim)
    ne = max(max(r[0], r[2]) for r in T) + 1
    nr = max(r[1] for r in T) + 1
    g = C.Lcg(seed)
    def draw(rows):
        return [[g.norm() for _ in range(d)] for _ in range(rows)]
    re_e = C.mat(re_e) if re_e is not None else draw(ne)
    im_e = C.mat(im_e) if im_e is not None else draw(ne)
    re_r = C.mat(re_r) if re_r is not None else draw(nr)
    im_r = C.mat(im_r) if im_r is not None else draw(nr)
    scores = []
    for h, r, t in T:
        s = 0.0
        for k in range(d):
            s += (re_e[h][k] * re_r[r][k] * re_e[t][k]
                  + re_e[h][k] * im_r[r][k] * im_e[t][k]
                  + im_e[h][k] * re_r[r][k] * im_e[t][k]
                  - im_e[h][k] * im_r[r][k] * re_e[t][k])
        scores.append(s)
    return RichResult(payload={
        "estimate": sum(scores) / len(scores), "scores": scores,
        "m": len(scores), "dim": d, "method": "ComplEx triple score"})


def cheatsheet():
    return "comple: ComplEx knowledge-graph embedding score."
