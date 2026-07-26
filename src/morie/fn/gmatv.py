# morie.fn -- function file (rootcoder007/morie)
"""Genomic relationship matrix (VanRaden 2008, methods 1 and 2)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["grm_vanraden"]


# Aliases follow VanRaden (2008) p.4416, verified against the primary paper.
# They were briefly wrong: an earlier pass took the numbering from the
# Montesinos secondary source, which renumbers VanRaden's methods. G_VR3 is
# deliberately absent -- VanRaden's third method regresses MM' on A and is not
# implemented here, so claiming the name would be a lie.
_METHOD_ALIASES = {
    1: 1, "G_VR1": 1,
    2: 2, "G_VR2": 2,
    3: 3, "G_XX": 3,
}


def grm_vanraden(markers, method=1):
    """Genomic relationship matrix per VanRaden (2008).

    Method numbering
    ----------------
    morie's integer ``method`` predates this audit and does NOT line up with
    the source's numbering. Both spellings are first-class; the string
    aliases follow the agrigenomic convention and are the unambiguous ones:

    ==============  ===========  =====================  =====================
    ``method=``     alias        VanRaden (2008)        Montesinos Sec 2.4
    ==============  ===========  =====================  =====================
    ``1`` (default) ``"G_VR1"``  Method 1, ZZ'/2sum-pq  Method 2
    ``2``           ``"G_VR2"``  Method 2, ZDZ'         Method 3
    ``3``           ``"G_XX"``   -- (not a VR method)   Method 1
    ==============  ===========  =====================  =====================

    The two sources number these differently, which is why the aliases carry
    the primary source's numbering and the table shows both. VanRaden's own
    Method 3 -- regressing MM' on A to get G = (MM' - g0*11')/g1, requiring no
    allele frequencies -- is NOT implemented, so there is deliberately no
    ``G_VR3`` alias. ``method=3`` is the uncentred XX'/m that the Montesinos
    chapter calls its Method 1; it is not one of VanRaden's three, hence
    ``G_XX`` rather than a ``G_VR*`` name.

    The integers are deliberately NOT renumbered to match the source: existing
    callers pass ``method=1`` and ``method=2``, and silently changing what
    those mean would leave working code returning a different matrix. Prefer
    the string aliases in new code.

    ``method=3`` / ``"G_XX"`` was previously unimplemented.

    Formulae
    --------
    ``"G_VR1"``  Z = M - P with column j of P equal to 2 p_j;
                 G = Z Z' / [2 * sum_j p_j (1 - p_j)]        (VanRaden Eq., p.4416)
    ``"G_VR2"``  G = Z D Z' with D diagonal, D_jj = 1 / (m * 2 p_j (1 - p_j));
                 equivalently z_ij = (x_ij - 2 p_j) / sqrt(2 p_j (1 - p_j))
                 and G = Z Z' / m                            (VanRaden, p.4416)
    ``"G_XX"``   G = X X' / m, uncentred  (Montesinos Sec 2.4 Method 1)

    Parameters
    ----------
    markers : array-like, shape (n, m)
        Genotype matrix, coded {0,1,2} (count of the reference allele).
    method : {1, 2, 3, "G_VR1", "G_VR2", "G_XX"}, default 1

    Returns
    -------
    RichResult with payload keys:
        estimate : (n,n) ndarray, the GRM
        diag_mean, off_mean : diagnostic averages
        p : allele frequencies, length m
        n, m : sample / marker counts
        method : description string

    References
    ----------
    VanRaden, P. M. (2008). Efficient methods to compute genomic predictions.
        *Journal of Dairy Science*, 91(11), 4414-4423, "Genomic Relationships
        and Inbreeding", p.4416. https://doi.org/10.3168/jds.2007-0980
        (PRIMARY -- now in the library.)
    Montesinos-Lopez et al., *Multivariate Statistical Machine Learning
        Methods for Genomic Prediction*, Sec. 2.4 "Methods to Compute the
        Genomic Relationship Matrix", pp. 49-52.

    Worked example: Montesinos-Lopez et al., Sec. 2.4, pp. 50-52 (8 lines x
    7 SNPs) -- transcribed in ``tests/fn/fixtures/gmatv.json``.

    Note: the secondary source contradicts itself twice, verified against the
    typeset PDF rather than the text extraction. For Method 1 (``"G_VR1"``)
    both the printed formula and the printed R code divide by the number of
    MARKERS (``dim(X)[2]``), but the printed numeric table is reproduced only
    by dividing by the number of LINES; we follow the formula, which is also
    VanRaden's canonical definition. For Method 3 (``"G_VR3"``) the printed
    formula uses the allele-frequency scaling ``sqrt(2p(1-p))`` while the
    printed R code uses ``scale(X, center=TRUE, scale=TRUE)`` -- the sample
    standard deviation -- and the printed table matches the R code, to a
    maximum absolute deviation of 0.162 from the formula. We follow the
    formula: allele-frequency scaling is the defining feature of VanRaden's
    Method 3, whereas sample-SD scaling yields standardised genotype scores,
    a different quantity. Both readings are recorded in the fixture.
    """
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2:
        raise ValueError("`markers` must be a 2D (n × m) array")
    try:
        mode = _METHOD_ALIASES[method]
    except (KeyError, TypeError):
        raise ValueError(
            "method must be one of: 1, 2, 3, 'G_VR1', 'G_VR2', 'G_XX' "
            f"(got {method!r})"
        ) from None
    n, m = M.shape
    # Allele frequencies (assume coding 0/1/2)
    p = M.mean(axis=0) / 2.0
    if mode == 3:
        # Method 1 of the source: uncentred, divided by the marker count.
        Z = M
        denom = float(m)
        method_str = "G_XX (uncentred XX'/m; Montesinos Method 1, not VanRaden)"
    elif mode == 2:
        Z = M - 2.0 * p
        scale = np.sqrt(2.0 * p * (1.0 - p))
        scale = np.where(scale > 0, scale, 1.0)
        Z = Z / scale
        denom = float(m)
        method_str = "VanRaden Method 2 / G_VR2 (ZDZ', per-locus scaled)"
    else:
        Z = M - 2.0 * p
        denom = float(2.0 * np.sum(p * (1.0 - p)))
        denom = denom if denom > 0 else 1.0
        method_str = "VanRaden Method 1 / G_VR1 (ZZ'/2sum-pq)"
    G = (Z @ Z.T) / denom
    diag_mean = float(np.mean(np.diag(G)))
    off = G - np.diag(np.diag(G))
    off_mean = float(np.sum(off) / (n * (n - 1))) if n > 1 else 0.0
    return RichResult(
        title="VanRaden Genomic Relationship Matrix",
        summary_lines=[
            ("n (individuals)", n),
            ("m (markers)", m),
            ("mean diag(G)", diag_mean),
            ("mean off-diag(G)", off_mean),
        ],
        payload={
            "estimate": G,
            "diag_mean": diag_mean,
            "off_mean": off_mean,
            "p": p,
            "n": n,
            "m": m,
            "method": method_str,
        },
    )


def cheatsheet():
    return "gmatv: Genomic relationship matrix (VanRaden G_VR1 / G_VR2, plus G_XX)"
