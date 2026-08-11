"""Getis-Ord Gi* hot/cold spot map with Benjamini-Hochberg FDR."""

from . import _array_core as np

from ._richresult import RichResult
from ._sci_core import erfc
from .getis import getis_ord_gi_star

__all__ = ["hotcld", "hot_cold_spots"]


def hotcld(x, W, alpha=0.05):
    """
    Hot/cold spot classification from Getis-Ord Gi* z-scores.

    Gi* follows the verified :func:`morie.fn.getis.getis_ord_gi_star`
    (sums include j = i; population sd; Getis & Ord 1992 eq. as
    corrected in Ord & Getis 1995, eq. 6-8). Two-sided normal p-values
    p_i = 2 (1 - Phi(|z_i|)) are then screened with the
    Benjamini-Hochberg step-up rule at level `alpha`:

        reject the k largest-ranked p_(i) with p_(k) <= k alpha / n.

    Classification: +1 (hot, z > 0 and BH-significant), -1 (cold,
    z < 0 and BH-significant), 0 otherwise.

    Sources
    -------
    Getis, A. & Ord, J. K. (1992). The analysis of spatial association by
    use of distance statistics. *Geographical Analysis*, 24(3), 189-206.
    Ord, J. K. & Getis, A. (1995). Local spatial autocorrelation
    statistics: distributional issues and an application. *Geographical
    Analysis*, 27(4), 286-306.
    Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery
    rate. *JRSS-B*, 57(1), 289-300 (step-up rule, Sec. 3, eq. before
    Thm. 1).
    Local reference implementation of the Gi* core: src/morie/fn/getis.py
    (verified; spdep::localGstar cross-checked in the parity tests).

    Parameters
    ----------
    x : array-like, (n,)
        Observed values.
    W : array-like, (n, n)
        Spatial weights; the diagonal may be nonzero (self-weights are
        part of Gi*).
    alpha : float
        FDR level for the BH screen.

    Returns
    -------
    RichResult
        Keys: z (Gi* z-scores), p (two-sided), significant (bool),
        category (+1/0/-1), n_hot, n_cold.
    """
    x = np.asarray(x, dtype=float).ravel()
    res = getis_ord_gi_star(x, W)
    z = np.asarray(res.local_values, dtype=float)
    n = z.size
    # two-sided normal p-value via erfc: 1 - Phi(t) = erfc(t/sqrt(2))/2
    p = np.asarray([float(erfc(abs(zi) / np.sqrt(2.0))) for zi in z])
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passing = np.where(ranked <= thresh)[0]
    significant = np.zeros(n, dtype=bool)
    if passing.size:
        k = int(passing[-1])
        significant[order[: k + 1]] = True
    category = np.where(significant & (z > 0), 1, np.where(significant & (z < 0), -1, 0))
    return RichResult(payload={
        "z": z, "p": p, "significant": significant, "category": category,
        "n_hot": int(np.sum(category == 1)), "n_cold": int(np.sum(category == -1)),
        "alpha": alpha, "n": int(n),
        "method": "Getis-Ord Gi* hot/cold spots, BH-FDR screened",
    })


# long descriptive alias (stub-era name)
hot_cold_spots = hotcld


def cheatsheet():
    return "hotcld: Getis-Ord Gi* z-scores + BH-FDR hot/cold classification"


# compact alias per ledger/NAMING.md
hotcoldspots = hot_cold_spots
