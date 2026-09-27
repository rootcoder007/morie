"""Spatial accessibility: two-step floating catchment (2SFCA, Gaussian 2SFCA), Hansen and gravity models.

Luo, W. and Wang, F. (2003). Measures of spatial accessibility to health care in a GIS
environment. Environment and Planning B 30, 865-884. Dai, D. (2010). Black residential
segregation, disparities in spatial access to health care facilities, and late-stage breast
cancer diagnosis in metropolitan Detroit. Health and Place 16, 1038-1052 (Gaussian 2SFCA).
Hansen, W. G. (1959). How accessibility shapes land use. Journal of the American Institute of
Planners 25, 73-76. Joseph, A. E. and Bantock, P. R. (1982). Measuring potential physical
accessibility to general practitioners in rural areas. Social Science and Medicine 16, 85-90.
"""

import math

from ._richresult import RichResult

__all__ = ["spatial_accessibility"]


def _ssum(it):
    s = 0.0
    for v in it:
        s += v
    return s


def spatial_accessibility(population, supply, dist, method="2sfca", d0=None, beta=1.0):
    r"""Accessibility A_i of demand locations i to supply sites j with capacities S_j.

    ``"2sfca"``: R_j = S_j / sum_{k: d_kj <= d0} P_k, A_i = sum_{j: d_ij <= d0} R_j.
    ``"g2sfca"``: the same with the Gaussian weight G(d) = (exp(-d^2 / (2 d0^2)) - exp(-1/2)) /
    (1 - exp(-1/2)) inside d0 (Dai 2010). ``"hansen"``: A_i = sum_j S_j exp(-beta d_ij).
    ``"gravity"``: A_i = sum_j S_j d_ij^-beta / V_j with V_j = sum_k P_k d_kj^-beta (Joseph and
    Bantock 1982). ``"nearest"``: distance to the closest site and the number within d0.

    Parameters
    ----------
    population : sequence
        Demand P_k per location.
    supply : sequence
        Capacity S_j per site.
    dist : matrix
        Demand-by-site distances.
    method : {"2sfca", "g2sfca", "hansen", "gravity", "nearest"}
    d0 : float, optional
        Catchment size (required by 2sfca, g2sfca, nearest).
    beta : float
        Decay for hansen and gravity.

    Returns
    -------
    RichResult
        Keys: accessibility (per demand location), ratio (R_j per site, for the
        catchment methods), nearest and within (for "nearest").

    References
    ----------
    Luo, W. and Wang, F. (2003). Environment and Planning B 30, 865-884.
    Dai, D. (2010). Health and Place 16, 1038-1052.
    Hansen, W. G. (1959). Journal of the American Institute of Planners 25, 73-76.

    Examples
    --------
    >>> spatial_accessibility([100, 100], [10], [[1.0], [3.0]], d0=2)["accessibility"]
    [0.1, 0.0]
    """
    P = [float(v) for v in population]
    S = [float(v) for v in supply]
    D = [[float(v) for v in r] for r in dist]
    n, m = len(P), len(S)
    if len(D) != n or any(len(r) != m for r in D):
        raise ValueError("dist must be len(population) x len(supply)")
    if method in ("2sfca", "g2sfca", "nearest") and d0 is None:
        raise ValueError(f'method "{method}" needs d0')
    out = {}
    if method in ("2sfca", "g2sfca"):
        e = math.exp(-0.5)

        def w(d):
            if d > d0:
                return 0.0
            return 1.0 if method == "2sfca" else (math.exp(-0.5 * (d / d0) ** 2) - e) / (1 - e)

        W = [[w(D[i][j]) for j in range(m)] for i in range(n)]
        R = []
        for j in range(m):
            den = _ssum(P[k] * W[k][j] for k in range(n))
            R.append(S[j] / den if den > 0 else 0.0)
        out["ratio"] = R
        out["accessibility"] = [_ssum(R[j] * W[i][j] for j in range(m)) for i in range(n)]
    elif method == "hansen":
        out["accessibility"] = [_ssum(S[j] * math.exp(-beta * D[i][j]) for j in range(m)) for i in range(n)]
    elif method == "gravity":
        V = [_ssum(P[k] * D[k][j] ** -beta for k in range(n)) for j in range(m)]
        out["ratio"] = [S[j] / V[j] for j in range(m)]
        out["accessibility"] = [_ssum(S[j] * D[i][j] ** -beta / V[j] for j in range(m)) for i in range(n)]
    elif method == "nearest":
        out["nearest"] = [min(r) for r in D]
        out["within"] = [sum(1 for v in r if v <= d0) for r in D]
        out["accessibility"] = out["nearest"]
    else:
        raise ValueError('method must be "2sfca", "g2sfca", "hansen", "gravity" or "nearest"')
    return RichResult(title="Spatial accessibility", summary_lines=[("method", method)], payload=out)


def cheatsheet():
    return "accidx: 2SFCA, Gaussian 2SFCA, Hansen, gravity and nearest-facility accessibility"
