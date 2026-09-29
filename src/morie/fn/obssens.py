# morie.fn -- function file (rootcoder007/morie)
"""Sensitivity analysis for matched observational studies (Rosenbaum, Design of Observational
Studies): amplification of Gamma into (Lambda, Delta), U-statistic signed-score sensitivity
bounds, the crosscut test for dose-response and its design sensitivity."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import normal_quantile

__all__ = [
    "gamma_from_lambda_delta",
    "amplify_gamma",
    "u_statistic_sensitivity",
    "crosscut_test",
    "crosscut_design_sensitivity",
]


def _pnorm(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _qnorm(p):
    return float(normal_quantile([p])[0])


def gamma_from_lambda_delta(lam, delta) -> RichResult:
    r"""Sensitivity parameter ``Gamma = (Lambda Delta + 1) / (Lambda + Delta)`` of a (Lambda, Delta) pair.

    ``Lambda`` bounds the odds of treatment within a pair and ``Delta`` the
    asymmetry of Wolfe's family ``Pr(W >= w | C) = omega Pr(W <= -w | C)``
    with ``1/Delta <= omega <= Delta`` (DOS eq. 3.28), which implies
    ``1 / (1 + Delta) <= Pr(W > 0 | C, |W| = w) <= Delta / (1 + Delta)``
    (eq. 3.29); a bias of ``(Lambda, Delta)`` has the same sensitivity
    bounds as ``Gamma`` (Rosenbaum and Silber 2009, Proposition 1).

    References
    ----------
    Rosenbaum, P. R. (2020). Design of Observational Studies, 2nd ed.,
    section 3.6. Rosenbaum, P. R. and Silber, J. H. (2009). Amplification of
    sensitivity analysis in matched observational studies. JASA 104, 1398-1405.

    Examples
    --------
    >>> r = gamma_from_lambda_delta(2.0, 3.0)
    >>> r.gamma, r.abz_bounds
    ([1.4], [[0.25, 0.75]])
    """
    lams = [float(v) for v in (lam if isinstance(lam, (list, tuple)) else [lam])]
    dels = [float(v) for v in (delta if isinstance(delta, (list, tuple)) else [delta])]
    if len(dels) == 1:
        dels = dels * len(lams)
    g = [(a * b + 1.0) / (a + b) for a, b in zip(lams, dels)]
    return RichResult(payload={"gamma": g, "abz_bounds": [[1.0 / (1.0 + b), b / (1.0 + b)] for b in dels]})


def amplify_gamma(gamma: float, lam) -> list:
    r"""Amplify a sensitivity parameter: the ``Delta`` paired with each ``Lambda > Gamma``.

    Solves ``Gamma = (Lambda Delta + 1) / (Lambda + Delta)`` for
    ``Delta = (Gamma Lambda - 1) / (Lambda - Gamma)`` (the curve of
    equivalent (Lambda, Delta) biases; as ``sensitivitymv::amplify``).

    References
    ----------
    Rosenbaum, P. R. and Silber, J. H. (2009). JASA 104, 1398-1405.

    Examples
    --------
    >>> [round(v, 12) for v in amplify_gamma(1.4, [2.0, 3.0])]
    [3.0, 2.0]
    """
    lams = [float(v) for v in (lam if isinstance(lam, (list, tuple)) else [lam])]
    return [(gamma * a - 1.0) / (a - gamma) if a > gamma else math.nan for a in lams]


def _avg_ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _choose(n, k):
    # product form n (n - 1) ... (n - k + 1) / k!, valid for real n (average ranks), as R's choose
    if k < 0:
        return 0.0
    out = 1.0
    for j in range(k):
        out *= (n - j) / (j + 1)
    return out


def _u_scores(rk, m, m1, m2, exact):
    n = len(rk)
    out = [0.0] * n
    for i, q in enumerate(rk):
        s = 0.0
        for ell in range(m1, m2 + 1):
            if exact:
                s += _choose(q - 1, ell - 1) / _choose(n, m) * _choose(n - q, m - ell)
            else:
                pk = q / n
                s += ell * _choose(m, ell) * pk ** (ell - 1) * (1 - pk) ** (m - ell)
        out[i] = s
    return out


def u_statistic_sensitivity(
    d,
    gamma: float = 1.0,
    *,
    m: int = 2,
    m1: int = 2,
    m2: int = 2,
    exact: bool | None = None,
    alternative: str = "greater",
) -> RichResult:
    r"""Upper bound on the one-sided P-value of Rosenbaum's U-statistic ``(m, m1, m2)`` under bias ``Gamma``.

    For matched-pair differences ``d`` with absolute ranks ``q_i`` (average
    ranks, zero differences scored 0), the scores are
    ``a_i = sum_{l=m1}^{m2} C(q_i - 1, l - 1) C(I - q_i, m - l) / C(I, m)``
    (exact, the default for ``I <= 50``) or their large-sample limit
    ``sum_l l C(m, l) p^(l-1) (1 - p)^(m-l)``, ``p = q_i / I``. With
    ``T = sum a_i 1(d_i > 0)`` and ``pi = Gamma / (1 + Gamma)``, the bound is
    ``1 - Phi((T - pi sum a_i) / sqrt(pi (1 - pi) sum a_i^2))``;
    ``(2, 2, 2)`` is Wilcoxon's signed rank, ``(m, m, m)`` Stephenson's test
    (DOS ch. 17 and 19; as ``DOS2::senU``).

    References
    ----------
    Rosenbaum, P. R. (2011). A new U-statistic with superior design
    sensitivity in matched observational studies. Biometrics 67, 1017-1027.
    Stephenson, W. R. (1981). A general class of one-sample nonparametric
    test statistics based on subsamples. JASA 76, 960-966.

    Examples
    --------
    >>> r = u_statistic_sensitivity([1.2, -0.3, 2.5, 0.8, 1.9, -0.6, 3.1, 0.4], 1.5, m=5, m1=4, m2=5)
    >>> round(r.p_value, 10)
    0.0546181559
    """
    n_pairs = len(d)
    if exact is None:
        exact = n_pairs <= 50
    pr = gamma / (1.0 + gamma)

    def dev(dd):
        ad = [abs(v) for v in dd]
        sc = _u_scores(_avg_ranks(ad), m, m1, m2, exact)
        sc = [s if a > 0 else 0.0 for s, a in zip(sc, ad)]
        ts = ssum(s for s, v in zip(sc, dd) if v > 0)
        ex = ssum(pr * s for s in sc)
        va = ssum(s * s for s in sc) * pr * (1 - pr)
        return (ts - ex) / math.sqrt(va), ts

    if alternative == "less":
        z, ts = dev([-v for v in d])
        p = 1.0 - _pnorm(z)
    elif alternative == "twosided":
        zl, _ = dev([-v for v in d])
        z, ts = dev(d)
        p = min(1.0, 2.0 * min(1.0 - _pnorm(zl), 1.0 - _pnorm(z)))
    else:
        z, ts = dev(d)
        p = 1.0 - _pnorm(z)
    return RichResult(payload={"p_value": p, "deviate": z, "statistic": ts})


def _q7(x, prob):
    s = sorted(float(v) for v in x)
    h = (len(s) - 1) * prob
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    w = h - lo
    return (1.0 - w) * s[lo] + w * s[hi] if w > 0 else s[lo]


def _fnch_upper(a, m1, m2, n, omega):
    lo, hi = max(0, n - m2), min(n, m1)
    lw = [
        math.lgamma(m1 + 1)
        - math.lgamma(x + 1)
        - math.lgamma(m1 - x + 1)
        + math.lgamma(m2 + 1)
        - math.lgamma(n - x + 1)
        - math.lgamma(m2 - n + x + 1)
        + x * math.log(omega)
        for x in range(lo, hi + 1)
    ]
    mx = max(lw)
    w = [math.exp(v - mx) for v in lw]
    tot = ssum(w)
    return ssum(w[x - lo] for x in range(max(a, lo), hi + 1)) / tot


def crosscut_test(x, y, ct: float = 0.25, gamma: float = 1.0) -> RichResult:
    r"""Crosscut test for a dose-response relationship and its sensitivity bound (Rosenbaum 2016).

    Pairs with dose ``x`` and outcome ``y`` both beyond their ``ct`` and
    ``1 - ct`` quantiles (type 7) form a 2 x 2 table of low/high dose by
    low/high response. The count ``A`` of low-low pairs is referred to
    Fisher's noncentral hypergeometric law with odds ``Gamma`` given the
    margins; the bound on the one-sided P-value is ``Pr(A' >= A)``
    (``Gamma = 1`` is the exact randomization test; ``ct = 1/2`` is the
    Olmstead-Tukey corner test).

    References
    ----------
    Rosenbaum, P. R. (2016). The crosscut statistic and its sensitivity to
    bias in observational studies with ordered doses of treatment.
    Biometrics 72, 175-183. Rosenbaum, P. R. (2020). Design of Observational
    Studies, 2nd ed., section 19.4.

    Examples
    --------
    >>> r = crosscut_test(list(range(20)), [v * 0.5 + (v % 3) for v in range(20)])
    >>> r.table, round(r.p_value, 10)
    ([[5, 0], [0, 4]], 0.0079365079)
    """
    qx1, qx2 = _q7(x, ct), _q7(x, 1 - ct)
    qy1, qy2 = _q7(y, ct), _q7(y, 1 - ct)
    tb = [[0, 0], [0, 0]]
    for a, b in zip(x, y):
        if (a <= qx1 or a >= qx2) and (b <= qy1 or b >= qy2):
            tb[1 if a >= qx2 else 0][1 if b >= qy2 else 0] += 1
    m1, m2 = tb[0][0] + tb[0][1], tb[1][0] + tb[1][1]
    n = tb[0][0] + tb[1][0]
    p = _fnch_upper(tb[0][0], m1, m2, n, gamma)
    return RichResult(payload={"table": tb, "p_value": p, "quantiles": [qx1, qx2, qy1, qy2]})


def _tanh_sinh(g, a, b, h=1.0 / 64.0, tmax=3.5):
    n = int(round(tmax / h))
    c, r = (a + b) / 2.0, (b - a) / 2.0
    s = 0.0
    for k in range(-n, n + 1):
        t = k * h
        v = math.pi / 2 * math.sinh(t)
        x = math.tanh(v)
        if abs(x) >= 1.0:
            continue
        s += g(c + r * x) * math.pi / 2 * math.cosh(t) / math.cosh(v) ** 2
    return s * h * r


def _bvn(h, k, rho):
    # P(X <= h, Y <= k) by Sheppard's formula, tanh-sinh quadrature in theta
    if rho == 0.0:
        return _pnorm(h) * _pnorm(k)
    b = math.asin(rho)

    def g(th):
        c = math.cos(th)
        return math.exp(-(h * h + k * k - 2 * h * k * math.sin(th)) / (2 * c * c))

    return _pnorm(h) * _pnorm(k) + _tanh_sinh(g, 0.0, b) / (2 * math.pi)


def crosscut_design_sensitivity(rho, eta) -> list:
    r"""Design sensitivity of the crosscut test for bivariate Normal dose and response.

    With ``z = Phi^(-1)(eta)`` the corner probabilities are
    ``p_LL = p_HH = Phi_2(z, z; rho)`` and ``p_LH = p_HL = eta - Phi_2(z, -z; rho)``,
    and the design sensitivity is the limiting corner odds ratio
    ``(p_LL p_HH) / (p_LH p_HL)`` (DOS Table 19.2). ``Phi_2`` is computed
    from Sheppard's integral by tanh-sinh quadrature.

    References
    ----------
    Rosenbaum, P. R. (2016). Biometrics 72, 175-183. Sheppard, W. F. (1900).
    On the calculation of the double integral expressing normal correlation.
    Trans. Cambridge Phil. Soc. 19, 23-66.

    Examples
    --------
    >>> [round(v, 1) for v in crosscut_design_sensitivity([0.3, 0.3], [0.5, 0.125])]
    [2.2, 32.1]
    """
    rs = [float(v) for v in (rho if isinstance(rho, (list, tuple)) else [rho])]
    es = [float(v) for v in (eta if isinstance(eta, (list, tuple)) else [eta])]
    out = []
    for r, e in zip(rs, es):
        z = _qnorm(e)
        same = _bvn(z, z, r)
        diff = e - _bvn(z, -z, r)
        out.append((same / diff) ** 2)
    return out


def cheatsheet() -> str:
    return (
        "gamma_from_lambda_delta / amplify_gamma / u_statistic_sensitivity / crosscut_test / "
        "crosscut_design_sensitivity -> Rosenbaum sensitivity analysis for matched observational studies."
    )
