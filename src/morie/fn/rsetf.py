# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ramsey RESET test for functional-form misspecification.

Ramsey JB (1969), *Tests for specification errors in classical linear
least-squares regression analysis*, Journal of the Royal Statistical
Society Series B 31(2):350-371.

This mirrors ``ramsey_reset_test`` in the R package's R/diagnostics.R,
so the two languages agree on the same design to full precision.
"""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["ramsey_reset"]

_METHOD = "Ramsey RESET test for functional-form misspecification"


def _gammaincc(a, x, itmax=400, eps=3e-14):
    """Regularised upper incomplete gamma Q(a, x)."""
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, dl = a, 1.0 / a, 1.0 / a
        for _ in range(itmax):
            ap += 1.0
            dl *= x / ap
            s += dl
            if abs(dl) < abs(s) * eps:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _betacf(a, b, x, itmax=300, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _f_upper_tail(f, df1, df2):
    """P(F_{df1, df2} >= f)."""
    if not np.isfinite(f) or f <= 0 or df1 <= 0 or df2 <= 0:
        return float("nan")
    return _betai(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))


def ramsey_reset(y, X, powers=(2, 3), add_intercept=False):
    r"""Test whether powers of the fitted values belong in the model.

    Fit :math:`y = X\beta + \epsilon`, take the fitted values
    :math:`\hat y`, and ask whether :math:`\hat y^2, \hat y^3, \ldots`
    add explanatory power:

    .. math::
        F = \frac{(\mathrm{SSR}_r - \mathrm{SSR}_u)/q}
                 {\mathrm{SSR}_u/(n - p - q)}.

    A large :math:`F` says the conditional mean is not linear in
    :math:`X`. What it does *not* say is which of the several possible
    reasons applies -- a missing quadratic term, a missing interaction,
    an omitted variable correlated with the fitted values, or a wrong
    link -- and the test cannot separate them, because every one of
    them shows up as curvature in :math:`\hat y`.

    Two properties worth keeping in view:

    **The powers are of the fitted values, not of the regressors.**
    That is what makes the test cheap and general: :math:`\hat y` is a
    single index summarising all of :math:`X`, so one auxiliary
    regression stands in for the whole space of polynomial and
    interaction terms. It is also why the test has no power against
    misspecification that leaves the index alone -- an omitted variable
    orthogonal to :math:`\hat y` passes cleanly.

    **:math:`\hat y` is estimated, and the test ignores that.** The
    auxiliary regressors are functions of :math:`\hat\beta`, not fixed
    covariates, so the :math:`F` statistic is exact only asymptotically.
    Ramsey's original derivation treats them as fixed; in finite
    samples the null distribution is slightly off, and the test is best
    read as approximate rather than exact.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design matrix. Include the intercept column yourself, or set
        ``add_intercept``.
    powers : sequence of int
        Powers of the fitted values to add. Ramsey's usual choice is
        (2, 3).
    add_intercept : bool
        Prepend a column of ones to ``X``.

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``df1``, ``df2``, ``ssr_restricted``,
        ``ssr_unrestricted``, ``r_squared_gain``, ``conclusion``.

    References
    ----------
    Ramsey JB (1969) *JRSS B* 31(2):350-371.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=400)
    >>> X = np.column_stack([np.ones(400), x])
    >>> lin = X @ [1.0, 2.0] + rng.normal(size=400) * 0.5
    >>> bool(ramsey_reset(lin, X)["p_value"] > 0.05)
    True
    >>> quad = lin + 1.5 * x ** 2
    >>> bool(ramsey_reset(quad, X)["p_value"] < 1e-6)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.shape[0] != yv.size and Xm.shape[1] == yv.size:
        Xm = Xm.T
    if add_intercept:
        Xm = np.column_stack([np.ones(Xm.shape[0]), Xm])
    n, p = Xm.shape
    if yv.size != n:
        raise ValueError(
            f"y has length {yv.size} but X has {n} rows."
        )
    pw = [int(v) for v in powers]
    if not pw:
        raise ValueError("powers must not be empty.")
    if any(v < 2 for v in pw):
        raise ValueError(
            f"powers must all be at least 2; got {pw}. The first power of "
            "the fitted values is collinear with X by construction."
        )
    q = len(pw)
    if n <= p + q:
        raise ValueError(
            f"need more than p + q = {p + q} observations; got {n}."
        )

    beta_r, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    yhat = Xm @ beta_r
    res_r = yv - yhat
    ssr_r = float(res_r @ res_r)

    # rescale the fitted values before taking powers: yhat^3 on a
    # response of order 1e3 overflows the conditioning of the normal
    # equations long before it overflows a double
    scale = float(np.max(np.abs(yhat)))
    scale = scale if scale > 0 else 1.0
    Z = np.column_stack([(yhat / scale) ** v for v in pw])
    Xu = np.column_stack([Xm, Z])
    beta_u, *_ = np.linalg.lstsq(Xu, yv, rcond=None)
    res_u = yv - Xu @ beta_u
    ssr_u = float(res_u @ res_u)

    df1, df2 = q, n - p - q
    num = (ssr_r - ssr_u) / df1
    den = ssr_u / df2
    f = num / den if den > 0 else float("inf")
    pval = _f_upper_tail(f, df1, df2)

    tss = float(np.sum((yv - yv.mean()) ** 2))
    gain = ((ssr_r - ssr_u) / tss) if tss > 0 else float("nan")

    reject = np.isfinite(pval) and pval < 0.05
    out = RichResult(
        title="Ramsey RESET test",
        summary_lines=[
            ("F", f),
            ("p-value", pval),
            ("df", f"({df1}, {df2})"),
            ("Powers", tuple(pw)),
        ],
        payload={
            "statistic": f,
            "estimate": f,
            "p_value": pval,
            "df1": df1,
            "df2": df2,
            "df": df1,
            "ssr_restricted": ssr_r,
            "ssr_unrestricted": ssr_u,
            "r_squared_gain": gain,
            "powers": tuple(pw),
            "fitted_scale": scale,
            "n": n,
            "n_params": p,
            "conclusion": (
                "Reject functional form (p < 0.05): consider nonlinear terms."
                if reject else "No evidence of misspecification."
            ),
            "method": _METHOD,
        },
        interpretation=(
            "Powers of the fitted values add significant explanatory power, "
            "so the conditional mean is not linear in X. The test does not "
            "say which of a missing polynomial term, a missing interaction "
            "or an omitted variable is responsible."
            if reject else
            "Powers of the fitted values add nothing significant. This is "
            "consistent with a correctly specified linear mean, but the test "
            "has no power against misspecification orthogonal to the fitted "
            "values."
        ),
    )
    if den <= 0:
        out.warnings.append(
            "The augmented model fits exactly, leaving no residual variance "
            "to test against. The F statistic is not defined."
        )
    return out


def cheatsheet():
    return (
        "rsetf: Ramsey RESET F test for functional-form misspecification, "
        "regressing on powers of the fitted values"
    )
