# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hardy-Weinberg equilibrium goodness-of-fit test.

Sources consulted: Hardy, G.H. (1908). Mendelian proportions in a mixed
population.  *Science* 28, 49-50; Weinberg, W. (1908). Ueber den Nachweis der
Vererbung beim Menschen.  *Jahreshefte des Vereins fuer vaterlaendische
Naturkunde in Wuerttemberg* 64, 369-382.  Both give the equilibrium genotype
proportions for a biallelic locus in a large randomly mating population,

    P(AA) = p^2,   P(Aa) = 2 p q,   P(aa) = q^2,   q = 1 - p

with the allele frequency estimated from the sample by gene counting,
p = (2 n_AA + n_Aa) / (2 N).  The departure from those proportions is
assessed with the standard Pearson goodness-of-fit statistic

    X^2 = sum (O - E)^2 / E

on one degree of freedom (three cells, one estimated parameter).  Hardy and
Weinberg state the proportions; the chi-square goodness-of-fit test around
them is the standard textbook application, not something either paper
derives.
"""

from __future__ import annotations

from . import _array_core as np

from . import t3util as _t3
from ._richresult import RichResult

__all__ = ["hardy_weinberg"]


def hardy_weinberg(genotypes):
    """Chi-square test of Hardy-Weinberg equilibrium.

    Parameters
    ----------
    genotypes : array-like
        Three observed genotype counts, ``(n_AA, n_Aa, n_aa)``.

    Returns
    -------
    RichResult
        statistic, p_value, df, p (allele frequency), expected, n, method.

    References
    ----------
    Hardy (1908), Science 28, 49-50; Weinberg (1908).
    """
    g = np.atleast_1d(np.asarray(genotypes, dtype=float)).ravel()
    n_aa = float(g[0])
    n_ab = float(g[1])
    n_bb = float(g[2])
    ntot = n_aa + n_ab + n_bb
    if ntot <= 0.0:
        return RichResult(
            payload={
                "statistic": float("nan"),
                "p_value": float("nan"),
                "df": 1,
                "p": float("nan"),
                "n": 0,
                "method": "Hardy-Weinberg equilibrium test (Hardy 1908; Weinberg 1908)",
            }
        )
    p = (2.0 * n_aa + n_ab) / (2.0 * ntot)
    q = 1.0 - p
    exp = [ntot * p * p, 2.0 * ntot * p * q, ntot * q * q]
    obs = [n_aa, n_ab, n_bb]
    stat = 0.0
    for i in range(3):
        if exp[i] > 0.0:
            stat += (obs[i] - exp[i]) ** 2 / exp[i]
    pval = _t3.chi2sf(stat, 1)
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(pval),
            "df": 1,
            "p": float(p),
            "q": float(q),
            "expected": np.asarray(exp, dtype=float),
            "n": int(ntot),
            "method": "Hardy-Weinberg equilibrium test (Hardy 1908; Weinberg 1908)",
        }
    )


# CANONICAL TEST
# >>> # counts exactly at equilibrium for p = 0.5: 25, 50, 25 -> X^2 = 0
# >>> r = hardy_weinberg([25.0, 50.0, 25.0])
# >>> assert abs(r["statistic"]) < 1e-12
# >>> assert abs(r["p"] - 0.5) < 1e-12


def cheatsheet():
    return "hwetst(counts): Hardy-Weinberg chi-square goodness-of-fit test."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
hardyweinberg = hardy_weinberg
