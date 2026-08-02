# morie.fn -- function file (rootcoder007/morie)
"""Astle-Balding genomic relationship matrix."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["astle_balding_grm"]


def astle_balding_grm(marker_matrix, freq=None):
    r"""Genomic relationship matrix with per-marker variance standardisation.

    .. math::
        G_{ij} = \frac{1}{m}\sum_{k=1}^{m}
            \frac{(x_{ik} - 2p_k)(x_{jk} - 2p_k)}{2p_k(1-p_k)},

    where :math:`x` counts reference alleles in {0, 1, 2} and :math:`p_k` is
    the allele frequency.

    The Astle-Balding scaling divides **each marker** by its own variance,
    unlike VanRaden's, which divides the total by :math:`\sum 2p(1-p)`. That
    difference is not cosmetic: dividing per marker gives every locus equal
    weight, so rare variants -- which have small :math:`2p(1-p)` -- are
    up-weighted substantially. That is desirable when rare variants carry real
    signal and harmful when they carry mostly genotyping error, which is why
    the choice between the two scalings is a modelling decision rather than a
    convention.

    Monomorphic markers have zero variance and would divide by zero; they are
    dropped and counted rather than silently producing infinities.

    Parameters
    ----------
    marker_matrix : array-like
        Genotypes ``(n, m)`` coded 0/1/2.
    freq : array-like, optional
        Allele frequencies. Estimated from the data otherwise, which makes the
        matrix sample-specific.

    Returns
    -------
    RichResult
        ``G``, ``n_markers_used``, ``n_dropped``, ``freq``,
        ``mean_diagonal``.

    References
    ----------
    Astle, W., & Balding, D. J. (2009). Population structure and cryptic
        relatedness in genetic association studies. *Statistical Science*,
        24(4), 451-471.

    Examples
    --------
    The matrix is symmetric with diagonal near 1 for unrelated individuals.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.15, 0.85, 400)
    >>> Xg = rng.binomial(2, p, size=(120, 400)).astype(float)
    >>> r = astle_balding_grm(Xg)
    >>> bool(np.allclose(r["G"], r["G"].T))
    True
    >>> bool(abs(r["mean_diagonal"] - 1.0) < 0.15)
    True

    Duplicated individuals show relatedness near 1 off-diagonal.

    >>> X2 = np.vstack([Xg, Xg[0]])
    >>> g = astle_balding_grm(X2)["G"]
    >>> bool(g[0, -1] > 0.8)
    True

    Monomorphic markers are dropped rather than dividing by zero.

    >>> Xm = np.c_[Xg, np.zeros(120)]
    >>> int(astle_balding_grm(Xm)["n_dropped"])
    1
    """
    X = np.atleast_2d(np.asarray(marker_matrix, dtype=float))
    n, m = X.shape
    if np.any((X < 0) | (X > 2)):
        raise ValueError("genotypes must be coded 0, 1 or 2")
    p = X.mean(axis=0) / 2.0 if freq is None else np.asarray(freq, dtype=float).ravel()
    if p.size != m:
        raise ValueError(f"freq has {p.size} entries but there are {m} markers")
    var = 2.0 * p * (1.0 - p)
    keep = var > 1e-12
    if not keep.any():
        raise ValueError("every marker is monomorphic")
    Z = (X[:, keep] - 2.0 * p[keep]) / np.sqrt(var[keep])
    G = Z @ Z.T / keep.sum()
    return RichResult(
        title="Astle-Balding GRM",
        summary_lines=[("individuals", int(n)), ("markers used", int(keep.sum())),
                       ("dropped", int((~keep).sum())),
                       ("mean diagonal", float(np.mean(np.diag(G))))],
        warnings=(["per-marker standardisation up-weights rare variants "
                   "relative to VanRaden scaling; that helps when rare variants "
                   "carry signal and hurts when they carry genotyping error"]
                  + ([] if freq is not None else
                     ["frequencies were estimated from this sample, so the "
                      "matrix is sample-specific and not comparable across "
                      "cohorts"])),
        payload={
            "G": G, "n_markers_used": int(keep.sum()),
            "n_dropped": int((~keep).sum()), "freq": p,
            "mean_diagonal": float(np.mean(np.diag(G))),
            "n": int(n), "method": "astle_balding_grm",
        },
    )


def cheatsheet():
    return "astlb: per-MARKER variance scaling up-weights rare variants (vs VanRaden's total scaling)"
