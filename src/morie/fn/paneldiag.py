# morie.fn -- function file (rootcoder007/morie)
"""Panel-data diagnostics: the within transformation, quadratic-form variance components,
cross-sectional dependence tests (Pesaran CD, Breusch-Pagan LM, scaled LM), Wooldridge's test
for unobserved effects, the Baltagi-Li LM test for AR(1)/MA(1) errors in the random-effects
model, the Breusch-Godfrey/Wooldridge serial-correlation test on within residuals and the
Conley spatial HAC covariance of OLS coefficients.

Rows are observations; ``unit`` gives the cross-section identifier and ``time`` the period.
Rows must be sorted by unit and, within a unit, by time (as ``plm`` sorts a ``pdata.frame``).
"""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rrng_core import pchisq, pf, pnorm

__all__ = [
    "panel_within",
    "panel_residuals",
    "panel_variance_components",
    "cross_section_dependence",
    "unobserved_effects_test",
    "baltagi_li_test",
    "panel_serial_test",
    "conley_vcov",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def _groups(unit):
    keys, idx = [], []
    pos = {}
    for u in list(unit):
        if u not in pos:
            pos[u] = len(keys)
            keys.append(u)
        idx.append(pos[u])
    return keys, idx


def _means(v, idx, g):
    s, c = [0.0] * g, [0] * g
    for x, i in zip(v, idx):
        s[i] += x
        c[i] += 1
    return [s[i] / c[i] for i in range(g)], c


def _demean(v, idx, g):
    m, _ = _means(v, idx, g)
    return [x - m[i] for x, i in zip(v, idx)]


def _ols(X, y):
    k = len(X[0])
    xtx = [[ssum(r[a] * r[b] for r in X) for b in range(k)] for a in range(k)]
    xty = [ssum(r[a] * v for r, v in zip(X, y)) for a in range(k)]
    beta = solve(xtx, xty)
    fit = [ssum(r[a] * beta[a] for a in range(k)) for r in X]
    return beta, [v - f for v, f in zip(y, fit)], fit


def panel_within(x, unit):
    r"""Within (fixed-effects) transformation ``x_it - xbar_i``.

    ``x`` is a vector or a matrix (one column per variable); each column is
    centred on its unit means.

    References
    ----------
    Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*, 6th edn.
    Springer, section 2.2.

    Examples
    --------
    >>> panel_within([1.0, 3.0, 2.0, 6.0], ["a", "a", "b", "b"])
    [-1.0, 1.0, -2.0, 2.0]
    """
    keys, idx = _groups(unit)
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        return _demean(_vec(x), idx, len(keys))
    M = _mat(x)
    cols = [_demean([r[j] for r in M], idx, len(keys)) for j in range(len(M[0]))]
    return [[c[i] for c in cols] for i in range(len(M))]


def panel_residuals(y, X, unit, model="pooling"):
    r"""Residuals of a pooled, within or unit-by-unit (heterogeneous) OLS panel regression.

    ``model="pooling"``: OLS of ``y`` on ``[1, X]``; ``"within"``: OLS of the
    within-transformed ``y`` on the within-transformed ``X`` (no intercept),
    returning the within residuals as ``plm`` does; ``"heterogeneous"``: a
    separate OLS of ``y`` on ``[1, X]`` for every unit.

    Examples
    --------
    >>> r = panel_residuals([1.0, 2.0, 2.0, 5.0], [[0.0], [1.0], [0.0], [1.0]], [1, 1, 2, 2], "within")
    >>> r.residuals, r.coefficients
    ([0.5, -0.5, -0.5, 0.5], [2.0])
    """
    yv = _vec(y)
    X = _mat(X)
    keys, idx = _groups(unit)
    g = len(keys)
    if model == "pooling":
        beta, res, _ = _ols([[1.0] + r for r in X], yv)
    elif model == "within":
        dy = _demean(yv, idx, g)
        cols = [_demean([r[j] for r in X], idx, g) for j in range(len(X[0]))]
        cols = [c for c in cols if max(abs(v) for v in c) > 0.0]
        beta, res, _ = _ols([[c[i] for c in cols] for i in range(len(yv))], dy)
    elif model == "heterogeneous":
        res = [0.0] * len(yv)
        beta = []
        for u in range(g):
            rows = [i for i in range(len(yv)) if idx[i] == u]
            b, e, _ = _ols([[1.0] + X[i] for i in rows], [yv[i] for i in rows])
            beta.append(b)
            for i, v in zip(rows, e):
                res[i] = v
    else:
        raise ValueError("model must be 'pooling', 'within' or 'heterogeneous'")
    return RichResult(payload={"residuals": res, "coefficients": beta, "model": model})


def panel_variance_components(resid, unit, df_correction=0):
    r"""Quadratic-form estimates of the one-way error-component variances (balanced panel).

    With ``u`` the residuals, ``sigma2_idios = u'Qu / (N(T-1) - df_correction)``,
    ``sigma2_1 = T sum_i ubar_i^2 / N`` and ``sigma2_id = (sigma2_1 -
    sigma2_idios) / T``; ``theta = 1 - sqrt(sigma2_idios / sigma2_1)`` is the
    random-effects GLS quasi-demeaning weight. Pooled OLS residuals give the
    Wallace-Hussain estimator; the residuals ``y - ybar - (X - Xbar) b_within``
    of the within slopes give the Amemiya estimator (``plm::ercomp`` methods
    ``"walhus"`` and ``"amemiya"``).

    References
    ----------
    Wallace, T. D. and Hussain, A. (1969). The use of error components models
    in combining cross section with time series data. *Econometrica* 37, 55-72.

    Amemiya, T. (1971). The estimation of the variances in a variance-components
    model. *International Economic Review* 12, 1-13.

    Examples
    --------
    >>> r = panel_variance_components([1.0, 0.0, 2.0, 1.5, -2.0, -1.0], [1, 1, 2, 2, 3, 3])
    >>> r.sigma2_idios, r.sigma2_1, r.sigma2_id
    (0.375, 3.7083333333333335, 1.6666666666666667)
    """
    u = _vec(resid)
    keys, idx = _groups(unit)
    g = len(keys)
    m, c = _means(u, idx, g)
    if min(c) != max(c):
        raise ValueError("panel_variance_components needs a balanced panel")
    T = c[0]
    q = ssum((v - m[i]) ** 2 for v, i in zip(u, idx))
    s2e = q / (g * (T - 1) - df_correction)
    s21 = T * ssum(v * v for v in m) / g
    return RichResult(
        payload={
            "sigma2_idios": s2e,
            "sigma2_1": s21,
            "sigma2_id": (s21 - s2e) / T,
            "theta": 1.0 - math.sqrt(s2e / s21),
            "n_units": g,
            "n_periods": T,
        }
    )


def cross_section_dependence(resid, unit, time, test="cd", w=None):
    r"""Tests for cross-sectional dependence of panel residuals.

    For every pair of units ``i > j`` with ``T_ij > 1`` common periods the
    residual correlation ``rho_ij`` is computed over the common periods.
    ``test="cd"`` (Pesaran 2004) ``CD = sqrt(1/m) sum sqrt(T_ij) rho_ij`` ~ N(0,1);
    ``"lm"`` (Breusch-Pagan 1980) ``sum T_ij rho_ij^2`` ~ chi^2 with ``m`` df;
    ``"sclm"`` the scaled LM ``sqrt(1/(2m)) sum (T_ij rho_ij^2 - 1)``;
    ``"bcsclm"`` its bias-corrected version (subtracting ``N / (2(T-1))``,
    for within residuals); ``"rho"`` and ``"absrho"`` the mean (absolute)
    correlation. ``m`` is the number of pairs used; a proximity matrix ``w``
    restricts the pairs to neighbours (local CD test).

    References
    ----------
    Pesaran, M. H. (2004). General diagnostic tests for cross section dependence
    in panels. CESifo Working Paper 1229.

    Breusch, T. S. and Pagan, A. R. (1980). The Lagrange multiplier test and its
    applications to model specification in econometrics. *Review of Economic
    Studies* 47, 239-253.

    Examples
    --------
    >>> e = [1.0, -1.0, 2.0, 1.0, -1.0, 3.0, -2.0, 1.0, 0.0]
    >>> r = cross_section_dependence(e, [1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3] * 3)
    >>> round(r.statistic, 12), r.n_pairs
    (0.154653670708, 3)
    """
    e = _vec(resid)
    keys, idx = _groups(unit)
    n = len(keys)
    series = [{} for _ in range(n)]
    for v, i, t in zip(e, idx, list(time)):
        series[i][t] = v
    W = None if w is None else _mat(w)
    terms = []
    tmax = 0
    for j in range(n):
        for i in range(j + 1, n):
            if W is not None and W[i][j] == 0:
                continue
            common = [t for t in series[i] if t in series[j]]
            T = len(common)
            tmax = max(tmax, T)
            if T <= 1:
                continue
            a = [series[i][t] for t in common]
            b = [series[j][t] for t in common]
            ma, mb = ssum(a) / T, ssum(b) / T
            sab = ssum((x - ma) * (y - mb) for x, y in zip(a, b))
            saa = ssum((x - ma) ** 2 for x in a)
            sbb = ssum((y - mb) ** 2 for y in b)
            terms.append((T, sab / math.sqrt(saa * sbb)))
    m = len(terms)
    if m == 0:
        raise ValueError("no pair of units with more than one common period")
    p = None
    if test == "cd":
        stat = math.sqrt(1.0 / m) * ssum(math.sqrt(T) * r for T, r in terms)
        p = 2.0 * pnorm(-abs(stat))
    elif test == "lm":
        stat = ssum(T * r * r for T, r in terms)
        p = pchisq(stat, m, lower_tail=False)
    elif test in ("sclm", "bcsclm"):
        stat = math.sqrt(1.0 / (2 * m)) * ssum(T * r * r - 1.0 for T, r in terms)
        if test == "bcsclm":
            stat -= n / (2.0 * (tmax - 1))
        p = 2.0 * pnorm(-abs(stat))
    elif test == "rho":
        stat = ssum(r for _, r in terms) / m
    elif test == "absrho":
        stat = ssum(abs(r) for _, r in terms) / m
    else:
        raise ValueError("test must be one of cd, lm, sclm, bcsclm, rho, absrho")
    return RichResult(payload={"statistic": stat, "p_value": p, "n_pairs": m, "test": test})


def unobserved_effects_test(resid, unit):
    r"""Wooldridge's semi-parametric test for unobserved individual effects.

    With pooled OLS residuals ``u``, ``S_i = sum_{t<s} u_it u_is`` and
    ``W = sum_i S_i / sqrt(sum_i S_i^2)``, asymptotically N(0,1) under no
    unobserved effect (``plm::pwtest``).

    References
    ----------
    Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel
    Data*, 2nd edn. MIT Press, section 10.4.4.

    Examples
    --------
    >>> r = unobserved_effects_test([1.0, 2.0, -1.0, -2.0, 0.5, -1.0], [1, 1, 2, 2, 3, 3])
    >>> round(r.statistic, 12)
    1.21854359169
    """
    u = _vec(resid)
    keys, idx = _groups(unit)
    S = []
    for g in range(len(keys)):
        v = [x for x, i in zip(u, idx) if i == g]
        s = 0.0
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                s += v[a] * v[b]
        S.append(s)
    z = ssum(S) / math.sqrt(ssum(s * s for s in S))
    return RichResult(payload={"statistic": z, "p_value": 2.0 * pnorm(-abs(z)), "S": S})


def _re_ml(yv, X, idx, g, T, tol=1e-14, maxit=10000):
    # Breusch (1987) fixed point for the ML variance ratio phi2 = sigma2_e / sigma2_1
    phi2 = 1.0
    for _ in range(maxit):
        theta = 1.0 - math.sqrt(phi2)
        my, _ = _means(yv, idx, g)
        mx = [_means([r[j] for r in X], idx, g)[0] for j in range(len(X[0]))]
        ys = [v - theta * my[i] for v, i in zip(yv, idx)]
        Xs = [[r[j] - theta * mx[j][i] for j in range(len(r))] for r, i in zip(X, idx)]
        beta, _, _ = _ols(Xs, ys)
        d = [v - ssum(r[a] * beta[a] for a in range(len(beta))) for v, r in zip(yv, X)]
        md, _ = _means(d, idx, g)
        q = ssum((v - md[i]) ** 2 for v, i in zip(d, idx))
        pq = T * ssum(v * v for v in md)
        new = min(1.0, q / ((T - 1) * pq))
        if abs(new - phi2) <= tol * phi2:
            phi2 = new
            break
        phi2 = new
    return beta, d, phi2


def baltagi_li_test(y, X, unit, alternative="twosided"):
    r"""Baltagi-Li LM test for AR(1)/MA(1) errors in the random-effects model (balanced).

    The random-intercept model is fitted by maximum likelihood (Breusch 1987
    iteration for ``sigma2_e / sigma2_1``); with its residuals ``u_i`` and
    ``E = I - J/T``, ``Jbar = J/T``, ``G`` the first-order adjacency matrix,
    ``sigma2_e = sum u_i'E u_i / (N(T-1))``, ``sigma2_1 = sum u_i'Jbar u_i / N``,
    ``D = N(T-1)/T (sigma2_1 - sigma2_e)/sigma2_1 + sigma2_e/2 sum u_i' A G A u_i``
    with ``A = Jbar/sigma2_1 + E/sigma2_e``, and ``J11`` the (rho, rho) element
    of the inverse information; ``LM = D^2 J11`` ~ chi^2_1 (two-sided) or
    ``D sqrt(J11)`` ~ N(0,1) (one-sided, ``plm::pbltest``).

    References
    ----------
    Baltagi, B. H. and Li, Q. (1995). Testing AR(1) against MA(1) disturbances
    in an error component model. *Journal of Econometrics* 68, 133-151.

    Breusch, T. S. (1987). Maximum likelihood estimation of random effects
    models. *Journal of Econometrics* 36, 383-389.

    Examples
    --------
    >>> y = [1.0, 2.0, 2.5, 4.0, 3.0, 3.5, 5.0, 6.5, 2.0, 1.0, 3.0, 2.0]
    >>> x = [[0.0], [1.0], [2.0], [3.0], [0.5], [1.0], [2.5], [3.0], [1.0], [0.0], [2.0], [1.5]]
    >>> r = baltagi_li_test(y, x, [1] * 4 + [2] * 4 + [3] * 4)
    >>> round(r.statistic, 10)
    0.2532721091
    """
    yv = _vec(y)
    X = [[1.0] + r for r in _mat(X)]
    keys, idx = _groups(unit)
    n = len(keys)
    _, c = _means(yv, idx, n)
    if min(c) != max(c):
        raise ValueError("baltagi_li_test needs a balanced panel")
    T = c[0]
    beta, u, phi2 = _re_ml(yv, X, idx, n, T)
    ui = [[v for v, i in zip(u, idx) if i == g] for g in range(n)]
    s2e = ssum(ssum((v - ssum(r) / T) ** 2 for v in r) for r in ui) / (n * (T - 1))
    s21 = ssum(T * (ssum(r) / T) ** 2 for r in ui) / n
    A = [[1.0 / (T * s21) + ((1.0 if a == b else 0.0) - 1.0 / T) / s2e for b in range(T)] for a in range(T)]
    G = [[1.0 if abs(a - b) == 1 else 0.0 for b in range(T)] for a in range(T)]
    AG = [[ssum(A[a][k] * G[k][b] for k in range(T)) for b in range(T)] for a in range(T)]
    S = [[ssum(AG[a][k] * A[k][b] for k in range(T)) for b in range(T)] for a in range(T)]
    star2 = ssum(ssum(r[a] * ssum(S[a][b] * r[b] for b in range(T)) for a in range(T)) for r in ui)
    D = (n * (T - 1) / T) * (s21 - s2e) / s21 + s2e / 2.0 * star2
    a = (s2e - s21) / (T * s21)
    j_rr = n * (2 * a * a * (T - 1) ** 2 + 2 * a * (2 * T - 3) + (T - 1))
    j12 = n * (T - 1) * s2e / s21**2
    j13 = n * (T - 1) / T * s2e * (1 / s21**2 - 1 / s2e**2)
    j22 = n * T * T / (2 * s21**2)
    j23 = n * T / (2 * s21**2)
    j33 = (n / 2) * (1 / s21**2 + (T - 1) / s2e**2)
    det = j_rr * (j22 * j33 - j23 * j23) - j12 * (j12 * j33 - j23 * j13) + j13 * (j12 * j23 - j22 * j13)
    J11 = n * n * T * T * (T - 1) / (det * 4 * s21 * s21 * s2e * s2e)
    if alternative == "onesided":
        stat = D * math.sqrt(J11)
        p = pnorm(stat, lower_tail=False)
    elif alternative == "twosided":
        stat = D * D * J11
        p = pchisq(stat, 1, lower_tail=False)
    else:
        raise ValueError("alternative must be 'twosided' or 'onesided'")
    return RichResult(
        payload={
            "statistic": stat,
            "p_value": p,
            "alternative": alternative,
            "coefficients": beta,
            "sigma2_e": s2e,
            "sigma2_1": s21,
            "phi2": phi2,
            "D": D,
            "J11": J11,
        }
    )


def panel_serial_test(y, X, unit, order=None, type="Chisq"):
    r"""Breusch-Godfrey/Wooldridge test for serial correlation in within (fixed-effects) models.

    The within-transformed ``y`` is regressed on the within-transformed ``X``
    (constant columns dropped); the stacked residuals ``e`` are regressed on
    the same regressors and ``order`` lags of ``e`` (lags filled with 0, taken
    along the stacked data as ``plm::pbgtest`` does). ``type="Chisq"``:
    ``n sum fitted^2 / sum e^2`` ~ chi^2 with ``order`` df; ``type="F"``: the
    F form. ``order`` defaults to the shortest unit length.

    References
    ----------
    Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models.
    *Australian Economic Papers* 17, 334-355.

    Godfrey, L. G. (1978). Testing against general autoregressive and moving
    average error models when the regressors include lagged dependent
    variables. *Econometrica* 46, 1293-1301.

    Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel
    Data*, 2nd edn. MIT Press, section 10.5.4.

    Examples
    --------
    >>> y = [1.0, 2.0, 2.5, 4.0, 3.0, 3.5, 5.0, 6.5, 2.0, 1.0, 3.0, 2.0]
    >>> x = [[0.0], [1.0], [2.0], [3.0], [0.5], [1.0], [2.5], [3.0], [1.0], [0.0], [2.0], [1.5]]
    >>> r = panel_serial_test(y, x, [1] * 4 + [2] * 4 + [3] * 4, order=1)
    >>> round(r.statistic, 10), r.df
    (0.0283140261, 1)
    """
    yv = _vec(y)
    X = _mat(X)
    keys, idx = _groups(unit)
    g = len(keys)
    _, c = _means(yv, idx, g)
    order = min(c) if order is None else int(order)
    dy = _demean(yv, idx, g)
    cols = [_demean([r[j] for r in X], idx, g) for j in range(len(X[0]))]
    cols = [cc for cc in cols if max(abs(v) for v in cc) > 0.0]
    n = len(yv)
    D = [[cc[i] for cc in cols] for i in range(n)]
    _, e, _ = _ols(D, dy)
    Z = [[e[i - L] if i - L >= 0 else 0.0 for L in range(1, order + 1)] for i in range(n)]
    _, ures, fit = _ols([D[i] + Z[i] for i in range(n)], e)
    k = len(cols)
    if type == "Chisq":
        stat = n * ssum(f * f for f in fit) / ssum(v * v for v in e)
        p = pchisq(stat, order, lower_tail=False)
        df = order
    elif type == "F":
        s0 = ssum(v * v for v in e)
        s1 = ssum(v * v for v in ures)
        stat = ((s0 - s1) / order) / (s1 / (n - k - order))
        df = (order, n - k - order)
        p = pf(stat, order, n - k - order, lower_tail=False)
    else:
        raise ValueError("type must be 'Chisq' or 'F'")
    return RichResult(payload={"statistic": stat, "p_value": p, "df": df, "order": order})


def _hav(la1, lo1, la2, lo2, radius):
    a = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(a)))


def _fx_pair(la1, lo1, la2, lo2, cutoff, distance):
    # fixest cpp_vcov_conley: pre-filters and distances with its constants (3.14159, 111 km, 12752 km)
    if abs(la2 - la1) > cutoff / 111 * 3.14159 / 180:
        return False
    dlon = abs(lo2 - lo1)
    dlon = dlon if dlon < 3.14159 else 6.28318 - dlon
    cm = math.cos((la1 + la2) / 2)
    if dlon > cutoff / 111 * 3.14159 / 180 / cm:
        return False
    if distance == "fixest_spherical":
        a = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
        return 12752 * math.asin(min(1.0, math.sqrt(a))) <= cutoff
    return (la2 - la1) ** 2 + (cm * dlon) ** 2 <= (cutoff * 3.14159 / 180 / 111) ** 2


def conley_vcov(X, resid, lat, lon, cutoff, kernel="uniform", distance="haversine", intercept=True, adjust=True):
    r"""Conley (1999) spatial HAC covariance of OLS coefficients.

    With scores ``s_i = x_i e_i`` and bread ``B = (X'X)^{-1}``,
    ``V = B (sum_i sum_j K(d_ij) s_i s_j') B``, times ``n/(n-k)`` when
    ``adjust``. ``K`` is the uniform kernel ``1(d <= cutoff)`` or the Bartlett
    kernel ``(1 - d/cutoff)_+``; ``d`` is the great-circle (haversine) distance
    in km on a sphere of radius 6371 km between points given in decimal
    degrees. ``distance="fixest_triangular"`` or ``"fixest_spherical"``
    reproduces ``fixest::vcov_conley`` (uniform kernel, its rounded constants
    3.14159, 111 km per degree and a 12752 km diameter).

    References
    ----------
    Conley, T. G. (1999). GMM estimation with cross sectional dependence.
    *Journal of Econometrics* 92, 1-45.

    Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with
    economic production in the Caribbean and Central America. *PNAS* 107,
    15367-15372.

    Examples
    --------
    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> r = conley_vcov(X, [0.5, -1.0, 1.0, -0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 5.0, 5.5], 100.0)
    >>> [round(v, 12) for v in r.se]
    [0.291547594742, 0.1]
    """
    Xm = _mat(X)
    if intercept:
        Xm = [[1.0] + r for r in Xm]
    e = _vec(resid)
    la = [v * math.pi / 180 for v in _vec(lat)]
    lo = [v * math.pi / 180 for v in _vec(lon)]
    n, k = len(Xm), len(Xm[0])
    s = [[r[a] * v for a in range(k)] for r, v in zip(Xm, e)]
    meat = [[ssum(s[i][a] * s[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    for i in range(n):
        for j in range(i + 1, n):
            if distance == "haversine":
                d = _hav(la[i], lo[i], la[j], lo[j], 6371.0)
                if kernel == "uniform":
                    wt = 1.0 if d <= cutoff else 0.0
                elif kernel == "bartlett":
                    wt = max(1.0 - d / cutoff, 0.0)
                else:
                    raise ValueError("kernel must be 'uniform' or 'bartlett'")
            elif distance in ("fixest_triangular", "fixest_spherical"):
                a, b = (i, j) if la[i] <= la[j] else (j, i)
                wt = 1.0 if _fx_pair(la[a], lo[a], la[b], lo[b], cutoff, distance) else 0.0
            else:
                raise ValueError("distance must be haversine, fixest_triangular or fixest_spherical")
            if wt != 0.0:
                for a in range(k):
                    for b in range(k):
                        meat[a][b] += wt * (s[i][a] * s[j][b] + s[j][a] * s[i][b])
    B = inverse([[ssum(r[a] * r[b] for r in Xm) for b in range(k)] for a in range(k)])
    BM = [[ssum(B[a][c] * meat[c][b] for c in range(k)) for b in range(k)] for a in range(k)]
    V = [[ssum(BM[a][c] * B[c][b] for c in range(k)) for b in range(k)] for a in range(k)]
    if adjust:
        V = [[v * n / (n - k) for v in r] for r in V]
    return RichResult(
        payload={
            "vcov": V,
            "se": [math.sqrt(V[a][a]) if V[a][a] >= 0 else math.nan for a in range(k)],
            "cutoff": cutoff,
        }
    )


def cheatsheet() -> str:
    return (
        "panel_within / panel_residuals / panel_variance_components / cross_section_dependence / "
        "unobserved_effects_test / baltagi_li_test / panel_serial_test / conley_vcov -> panel diagnostics."
    )
