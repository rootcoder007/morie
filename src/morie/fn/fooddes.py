"""USDA low-income, low-access (LILA) food desert classification.

Dutko, P., Ver Ploeg, M. and Farrigan, T. (2012). Characteristics and Influential Factors of Food
Deserts. USDA Economic Research Service, Economic Research Report 140. Rhone, A. et al. (2017).
Low-Income and Low-Supermarket-Access Census Tracts, 2010-2015. USDA ERS, Economic
Information Bulletin 165.
"""

from ._richresult import RichResult

__all__ = ["food_desert"]


def food_desert(population, beyond, poverty_rate, family_income, area_median_income, urban, share=1 / 3, count=500):
    r"""Flag census tracts as low income, low access and LILA food deserts.

    Low income: poverty rate >= 20 %, or median family income <= 80 % of the area median
    (the New Markets Tax Credit definition). Low access: at least ``count`` people or at
    least ``share`` of the tract population live beyond the distance threshold from the
    nearest supermarket (1 mile urban, 10 miles rural; ``beyond`` gives that population per
    tract, measured with the urban/rural threshold). LILA = low income and low access.

    Parameters
    ----------
    population, beyond : sequences
        Tract population and the part of it beyond the access threshold.
    poverty_rate : sequence
        Proportion in poverty (0-1).
    family_income, area_median_income : sequences
        Tract median family income and the relevant area (state or metro) median.
    urban : sequence of bool
        Only used to report which threshold applied.
    share, count : float, int
        Low-access thresholds.

    Returns
    -------
    RichResult
        Keys: low_income, low_access, lila, threshold_miles, n_lila, lila_population.

    References
    ----------
    Dutko, P., Ver Ploeg, M. and Farrigan, T. (2012). USDA ERS Economic Research Report 140.
    Rhone, A. et al. (2017). USDA ERS Economic Information Bulletin 165.

    Examples
    --------
    >>> food_desert([4000], [1500], [0.25], [40000], [60000], [True])["lila"]
    [True]
    """
    n = len(population)
    li = [
        float(poverty_rate[i]) >= 0.20 or float(family_income[i]) <= 0.80 * float(area_median_income[i])
        for i in range(n)
    ]
    la = [
        float(beyond[i]) >= count or (float(population[i]) > 0 and float(beyond[i]) / float(population[i]) >= share)
        for i in range(n)
    ]
    lila = [a and b for a, b in zip(li, la)]
    pop = 0.0
    for i in range(n):
        if lila[i]:
            pop += float(population[i])
    return RichResult(
        title="Food desert (LILA) classification",
        summary_lines=[("LILA tracts", sum(lila))],
        payload={
            "low_income": li,
            "low_access": la,
            "lila": lila,
            "threshold_miles": [1.0 if u else 10.0 for u in urban],
            "n_lila": sum(lila),
            "lila_population": pop,
        },
    )


def cheatsheet():
    return "fooddes: USDA low-income low-access (LILA) food desert flags"
