"""Polarization measures: Esteban-Ray, Foster-Wolfson bipolarization, party sorting, affective
polarization, group separation and its decomposition by dimension.

Esteban, J.-M. and Ray, D. (1994). On the measurement of polarization. Econometrica 62, 819-851.
Wolfson, M. C. (1994). When inequalities diverge. American Economic Review 84(2), 353-358.
Foster, J. E. and Wolfson, M. C. (2010). Polarization and the decline of the middle class:
Canada and the U.S. Journal of Economic Inequality 8, 247-273. Levendusky, M. (2009). The
Partisan Sort. University of Chicago Press. Iyengar, S., Sood, G. and Lelkes, Y. (2012). Affect,
not ideology: a social identity perspective on polarization. Public Opinion Quarterly 76,
405-431.
"""

import math

from ._richresult import RichResult

__all__ = ["polarization_measures"]


def _s(it):
    s = 0.0
    for v in it:
        s += v
    return s


def polarization_measures(x, party=None, alpha=1.0, weights=None, in_rating=None, out_rating=None):
    r"""Polarization of positions ``x`` (one value per person, or points for the dimensional measures).

    - Esteban-Ray: ER = sum_i sum_j p_i^(1 + alpha) p_j |y_i - y_j| over the distinct values y with
      population shares p (alpha in (0, 1.6]; Esteban and Ray 1994, K = 1).
    - Foster-Wolfson bipolarization: W = 2 (2 T - G) mu / m, T = 1/2 - L(1/2) the gap between the
      bottom half's Lorenz share and one half, G the Gini, mu the mean, m the median (positive x).
    - With a binary ``party``: sorting = Pearson correlation of party with x (Levendusky 2009);
      separation = |mean_1 - mean_0| / pooled SD in 1D, the Mahalanobis distance of the party
      centroids under the pooled covariance for points; and a per-dimension decomposition of the
      total sum of squares into between-party and within-party parts.
    - With ``in_rating``/``out_rating`` (feeling thermometers): affective polarization = mean
      in-party minus mean out-party rating (Iyengar et al. 2012).

    Parameters
    ----------
    x : sequence of numbers, or list of points
    party : sequence of 0/1, optional
    alpha : float
    weights : sequence, optional
        Person weights for Esteban-Ray (Foster-Wolfson is unweighted).
    in_rating, out_rating : sequences, optional

    Returns
    -------
    RichResult
        Keys (as applicable): esteban_ray, wolfson, sorting, separation, between_share,
        affective.

    References
    ----------
    Esteban, J.-M. and Ray, D. (1994). Econometrica 62, 819-851.
    Foster, J. E. and Wolfson, M. C. (2010). Journal of Economic Inequality 8, 247-273.
    Levendusky, M. (2009). The Partisan Sort.
    Iyengar, S., Sood, G. and Lelkes, Y. (2012). Public Opinion Quarterly 76, 405-431.

    Examples
    --------
    >>> polarization_measures([0, 0, 1, 1])["esteban_ray"]
    0.25
    """
    pts = isinstance(x[0], (list, tuple))
    out = {}
    if not pts:
        v = [float(t) for t in x]
        n = len(v)
        w = [1.0] * n if weights is None else [float(t) for t in weights]
        W = _s(w)
        shares = {}
        for t, wi in zip(v, w):
            shares[t] = shares.get(t, 0.0) + wi / W
        ys = sorted(shares)
        out["esteban_ray"] = _s(shares[a] ** (1 + alpha) * shares[b] * abs(a - b) for a in ys for b in ys)
        if min(v) > 0:
            sv = sorted(v)
            mu = _s(sv) / n
            med = sv[n // 2] if n % 2 else 0.5 * (sv[n // 2 - 1] + sv[n // 2])
            low = _s(sv[: n // 2]) + (0.5 * sv[n // 2] if n % 2 else 0.0)  # bottom half of the population
            L_half = low / (mu * n)
            G = _s(abs(p - q) for p in sv for q in sv) / (2 * n * n * mu)
            out["wolfson"] = 2 * (2 * (0.5 - L_half) - G) * mu / med
            out["gini"] = G
    if party is not None:
        g = [int(t) for t in party]
        P = [[float(t) for t in r] for r in x] if pts else [[float(t)] for t in x]
        d = len(P[0])
        n = len(P)
        idx = {0: [i for i in range(n) if g[i] == 0], 1: [i for i in range(n) if g[i] == 1]}
        mean = {k: [_s(P[i][j] for i in idx[k]) / len(idx[k]) for j in range(d)] for k in (0, 1)}
        grand = [_s(P[i][j] for i in range(n)) / n for j in range(d)]
        S = [
            [_s((P[i][a] - mean[g[i]][a]) * (P[i][b] - mean[g[i]][b]) for i in range(n)) / (n - 2) for b in range(d)]
            for a in range(d)
        ]
        diff = [mean[1][j] - mean[0][j] for j in range(d)]
        if d == 1:
            out["separation"] = abs(diff[0]) / math.sqrt(S[0][0])
            xs = [p[0] for p in P]
            mx, mg = _s(xs) / n, _s(g) / n
            cov = _s((xs[i] - mx) * (g[i] - mg) for i in range(n))
            out["sorting"] = cov / math.sqrt(_s((t - mx) ** 2 for t in xs) * _s((t - mg) ** 2 for t in g))
        else:
            det = S[0][0] * S[1][1] - S[0][1] * S[1][0] if d == 2 else None
            if d != 2:
                raise ValueError("dimensional separation is implemented for one or two dimensions")
            inv = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
            out["separation"] = math.sqrt(_s(diff[a] * inv[a][b] * diff[b] for a in range(2) for b in range(2)))
        tss = [_s((P[i][j] - grand[j]) ** 2 for i in range(n)) for j in range(d)]
        bss = [_s(len(idx[k]) * (mean[k][j] - grand[j]) ** 2 for k in (0, 1)) for j in range(d)]
        out["between_share"] = [b / t if t > 0 else 0.0 for b, t in zip(bss, tss)]
    if in_rating is not None and out_rating is not None:
        a = [float(t) for t in in_rating]
        b = [float(t) for t in out_rating]
        out["affective"] = _s(a) / len(a) - _s(b) / len(b)
    return RichResult(title="Polarization measures", summary_lines=[("measures", ", ".join(out))], payload=out)


def cheatsheet():
    return "svpolr: Esteban-Ray, Foster-Wolfson, party sorting, affective polarization, separation"
