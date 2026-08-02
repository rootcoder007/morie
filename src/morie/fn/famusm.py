# morie.fn -- function file (rootcoder007/morie)
"""Transmission disequilibrium test (Spielman, McGinnis & Ewens 1993)."""

from __future__ import annotations

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["family_based_assoc"]


def family_based_assoc(trios):
    r"""TDT for family-based association from transmitted/untransmitted counts.

    Among heterozygous parents, count transmissions of the marker allele
    (:math:`b`) against non-transmissions (:math:`c`). Under the null of no
    linkage *or* no association, each is equally likely, so

    .. math::

        \chi^2_{TDT} = \frac{(b-c)^2}{b+c} \sim \chi^2_1 .

    Parameters
    ----------
    trios : array-like
        Either the two counts ``(b, c)``, or a 2-D array of per-trio
        transmission indicators with columns ``(transmitted, untransmitted)``
        which are summed. Counts must be non-negative integers.

    Returns
    -------
    RichResult
        keys: ``estimate`` (:math:`\chi^2`), ``statistic``, ``p_value``,
        ``b``, ``c``, ``n_informative``, ``odds_ratio``, ``df``, ``method``.

    Raises
    ------
    ValueError
        If the input is not interpretable as ``(b, c)``, if any count is
        negative or non-integral, or if ``b + c == 0``.

    References
    ----------
    Spielman, R. S., McGinnis, R. E., & Ewens, W. J. (1993). Transmission
        test for linkage disequilibrium: the insulin gene region and
        insulin-dependent diabetes mellitus (IDDM). *American Journal of
        Human Genetics*, 52(3), 506-516.

    Notes
    -----
    **Only heterozygous parents are informative.** A homozygous parent
    transmits the same allele whatever happens, so it contributes no evidence
    and must not enter :math:`b` or :math:`c`; including homozygous
    transmissions inflates :math:`b + c` and drives :math:`\chi^2` toward
    zero. This function receives the counts already restricted to
    heterozygous parents and cannot check that for you.

    This is McNemar's test on the transmitted/untransmitted pair. The
    :math:`\chi^2_1` reference is asymptotic; with :math:`b + c` small
    (conventionally under about 10) the exact binomial is preferable, and the
    returned ``n_informative`` is what to check that against.

    The point of the TDT is that it is immune to population stratification --
    a case-control association can be entirely an artefact of ancestry, while
    the transmission comparison is internal to each family.
    """
    arr = np.asarray(trios, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        b, c = float(arr[:, 0].sum()), float(arr[:, 1].sum())
    elif arr.ndim == 1 and arr.size == 2:
        b, c = float(arr[0]), float(arr[1])
    else:
        raise ValueError(
            f"trios must be either the two counts (b, c) or an (n, 2) array of "
            f"per-trio (transmitted, untransmitted) indicators; got shape {arr.shape}"
        )
    if b < 0 or c < 0:
        raise ValueError(f"counts must be non-negative; got b={b!r}, c={c!r}")
    if b != int(b) or c != int(c):
        raise ValueError(f"counts must be whole numbers; got b={b!r}, c={c!r}")
    n_inf = b + c
    if n_inf == 0:
        raise ValueError(
            "b + c == 0: no heterozygous (informative) parents, so the TDT is "
            "undefined. Homozygous parents carry no transmission information."
        )
    chi2 = (b - c) ** 2 / n_inf
    p = float(stats.chi2.sf(chi2, 1))
    odds = (b / c) if c > 0 else float("inf")
    return RichResult(
        payload={
            "estimate": float(chi2),
            "statistic": float(chi2),
            "p_value": p,
            "b": int(b),
            "c": int(c),
            "n_informative": int(n_inf),
            "odds_ratio": odds,
            "df": 1,
            "method": "transmission disequilibrium test (Spielman, McGinnis & Ewens 1993)",
        }
    )


def cheatsheet():
    return "famusm: TDT chi2 = (b-c)^2/(b+c), df=1 (Spielman-McGinnis-Ewens 1993)."
