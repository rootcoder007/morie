# morie.fn -- slice s04 (rootcoder007/morie)
"""Narrow-sense heritability from LMM variance components.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 141-170], Chapter 5, Section 5.3,
equation (5.3).  That equation writes the genomic mixed model as
Y = 1_n mu + Z_L b + e with b ~ N_J(0, sigma_g^2 G) and R = sigma^2 I_n,
so the phenotypic variance of a line is sigma_g^2 + sigma^2.

NOT IN THE BOOK.  The whole of the book was searched (all seventeen
page-range volumes and the index, [Pages 683-691]); "heritability"
appears only in prose and the index has no entry for it.  The book never
writes the ratio down.  The ratio itself,

    h^2 = sigma_g^2 / (sigma_g^2 + sigma_e^2),

is taken from de los Campos, G., Sorensen, D. and Gianola, D. (2015).
Genomic heritability: what is it?  *PLoS Genetics* 11(5), e1005048,
which defines it as the genetic share of the phenotypic variance of
exactly the model (5.3) decomposition.  The variance decomposition is
the book's; the ratio is de los Campos et al.'s.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["heritability_lmm"]


def heritability_lmm(sigma_g2, sigma_e2):
    """Narrow-sense heritability from the two LMM variance components.

    Parameters
    ----------
    sigma_g2 : array-like
        Genetic variance component(s), non-negative.
    sigma_e2 : array-like
        Residual variance component(s), non-negative; recycled against
        sigma_g2 when one of the two is a single value.

    Returns
    -------
    estimate : h^2 for the first pair
    h2       : the vector of heritabilities
    sigma_p2 : the phenotypic variances sigma_g^2 + sigma_e^2
    """
    g = k.vec(sigma_g2)
    e = k.vec(sigma_e2)
    if not g or not e:
        raise ValueError("heritability_lmm: both variance components are required")
    if len(g) != len(e) and len(g) != 1 and len(e) != 1:
        raise ValueError("heritability_lmm: sigma_g2 and sigma_e2 have incompatible lengths")
    n = max(len(g), len(e))
    h2 = []
    sp = []
    for i in range(n):
        a = g[i % len(g)]
        b = e[i % len(e)]
        if a < 0.0 or b < 0.0:
            raise ValueError("heritability_lmm: variance components must be non-negative")
        p = a + b
        if p <= 0.0:
            raise ValueError("heritability_lmm: phenotypic variance is zero")
        sp.append(p)
        h2.append(a / p)
    return RichResult(
        title="Narrow-sense heritability",
        summary_lines=[("components", n)],
        payload={
            "estimate": h2[0],
            "h2": h2,
            "sigma_p2": sp,
            "n": n,
            "method": "h2 = sigma_g^2/(sigma_g^2+sigma_e^2); Ch 5 eq. (5.3) decomposition, ratio from de los Campos et al. (2015)",
        },
    )


def cheatsheet():
    return "h2est: Narrow-sense heritability from LMM variance components"
