"""Service coverage of demand by facilities within a standard distance (Toregas and ReVelle 1973; Daskin and Stern 1981).

Daskin, M. S. and Stern, E. H. (1981). A hierarchical objective set covering model for
emergency medical service vehicle deployment. Transportation Science 15, 137-152.
"""

from ._richresult import RichResult

__all__ = ["service_coverage"]


def service_coverage(dist, radius, weights=None):
    r"""Coverage summary for demand points given demand-by-facility distances.

    A point is covered when some facility lies within ``radius``; its coverage multiplicity
    (backup coverage, Daskin and Stern 1981) is the number of such facilities. Reports the
    covered weight and share, the weighted mean nearest distance, the uncovered points and
    the demand weight each facility covers.

    Parameters
    ----------
    dist : matrix (demand x facilities)
    radius : float
    weights : sequence, optional
        Demand weights (population).

    Returns
    -------
    RichResult
        Keys: covered, multiplicity, nearest, covered_weight, coverage_share,
        mean_nearest, uncovered, facility_load.

    References
    ----------
    Daskin, M. S. and Stern, E. H. (1981). Transportation Science 15, 137-152.

    Examples
    --------
    >>> service_coverage([[1.0, 5.0], [4.0, 2.5], [6.0, 7.0]], 3.0, [10, 20, 30])["coverage_share"]
    0.5
    """
    D = [[float(v) for v in r] for r in dist]
    n = len(D)
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    R = float(radius)
    mult = [sum(1 for v in r if v <= R) for r in D]
    near = [min(r) for r in D]
    covw = 0.0
    tot = 0.0
    mn = 0.0
    for i in range(n):
        tot += w[i]
        mn += w[i] * near[i]
        if mult[i]:
            covw += w[i]
    load = []
    for j in range(len(D[0])):
        s = 0.0
        for i in range(n):
            if D[i][j] <= R:
                s += w[i]
        load.append(s)
    return RichResult(
        title="Service coverage",
        summary_lines=[("coverage share", covw / tot)],
        payload={
            "covered": [m > 0 for m in mult],
            "multiplicity": mult,
            "nearest": near,
            "covered_weight": covw,
            "coverage_share": covw / tot,
            "mean_nearest": mn / tot,
            "uncovered": [i for i in range(n) if not mult[i]],
            "facility_load": load,
        },
    )


def cheatsheet():
    return "servcov: facility service coverage within a standard distance, with backup multiplicity"
