# morie.fn -- shelf core (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Modern time-series forecasting shelf core.

Spec: Joseph, M. and Tackes, J. (2024), *Modern Time Series
Forecasting with Python*, 2nd ed., Packt.  Locators are the printed
page numbers of that edition.

Sourcing rule applied here.  Where the book prints the formula, the
formula is quoted from the book and the page given.  Where the book
only NAMES a method and points at its paper -- which is the case for
every deep architecture in the shelf -- the equations are taken from
the paper itself, fetched and quoted, and the docstring names the
paper, the arXiv id and the printed equation numbers.  Nothing is
reconstructed from memory.

Determinism.  Every learned weight is CALLER-SUPPLIED; no layer here
initializes anything at random, no split shuffles, no bootstrap
resamples.  The R mirror in R/ts_joseph.R therefore reproduces each
number to machine precision.
"""

from __future__ import annotations

import math

_TWOPI = 2.0 * math.pi


def _vec(x, name="x"):
    v = [float(t) for t in x]
    if not v:
        raise ValueError("%s must be non-empty" % name)
    return v


def _pair(y, yhat):
    a = _vec(y, "y")
    b = _vec(yhat, "yhat")
    if len(a) != len(b):
        raise ValueError("y and yhat must be the same length")
    return a, b


def _mean(v):
    return sum(v) / len(v)


def _solve(a, b):
    """Gaussian elimination with partial pivoting; deterministic."""
    n = len(a)
    m = [list(map(float, a[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: (abs(m[r][c]), -r))
        if abs(m[piv][c]) < 1e-300:
            raise ValueError("singular system")
        m[c], m[piv] = m[piv], m[c]
        pv = m[c][c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / pv
            if f == 0.0:
                continue
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _ols(x, y):
    """Least squares by normal equations; x is a list of rows."""
    n = len(x)
    p = len(x[0])
    xtx = [[sum(x[i][a] * x[i][b] for i in range(n)) for b in range(p)]
           for a in range(p)]
    xty = [sum(x[i][a] * y[i] for i in range(n)) for a in range(p)]
    for a in range(p):
        xtx[a][a] += 1e-12
    return _solve(xtx, xty)


# =====================================================================
# Forecast error metrics -- ch. 19, "Evaluating Forecast Errors"
# =====================================================================

def rmse(y, yhat):
    """Root mean squared error, ch. 19 p. 566."""
    a, b = _pair(y, yhat)
    e = [a[i] - b[i] for i in range(len(a))]
    mse = sum(v * v for v in e) / len(e)
    return {"rmse": math.sqrt(mse), "mse": mse, "mae": sum(abs(v) for v in e) / len(e),
            "bias": _mean(e), "n": len(e)}


def mapets(y, yhat):
    """Mean absolute percentage error, ch. 19 p. 568.

    The book's own warning is honoured: MAPE "breaks down when the
    actual observation is zero (due to division by zero)" (p. 568), so
    zero actuals raise rather than silently returning infinity.
    """
    a, b = _pair(y, yhat)
    if any(v == 0.0 for v in a):
        raise ValueError("MAPE is undefined when an actual value is zero")
    pe = [100.0 * abs(a[i] - b[i]) / abs(a[i]) for i in range(len(a))]
    srt = sorted(pe)
    mid = len(srt) // 2
    med = srt[mid] if len(srt) % 2 else 0.5 * (srt[mid - 1] + srt[mid])
    return {"mape": _mean(pe), "mdape": med, "maxape": max(pe), "n": len(pe)}


def smape(y, yhat):
    """Symmetric MAPE, ch. 19 p. 569.

    Quoted from p. 569:
        sMAPE = (1/H) sum_t 200 |e_t| / (|y_t| + |yhat_t|)

    Note the 200 in the numerator, which is the book's own convention:
    the symmetric denominator is the SUM of the magnitudes, not their
    average, so the factor is 200 rather than 100.
    """
    a, b = _pair(y, yhat)
    terms = []
    for i in range(len(a)):
        den = abs(a[i]) + abs(b[i])
        if den == 0.0:
            raise ValueError("sMAPE is undefined when |y| + |yhat| is zero")
        terms.append(200.0 * abs(a[i] - b[i]) / den)
    srt = sorted(terms)
    mid = len(srt) // 2
    med = srt[mid] if len(srt) % 2 else 0.5 * (srt[mid - 1] + srt[mid])
    return {"smape": _mean(terms), "smdape": med, "n": len(terms)}


def rmsse(y, yhat, insample, season=1):
    """Root mean squared scaled error, ch. 19 p. 572.

    Quoted from p. 572: the squared errors are scaled by the in-sample
    mean squared error of the naive forecast,

        RMSSE = sqrt( (1/H) sum_t e_t^2
                      / ( (1/(T-1)) sum_{i=2..T} (y_i - y_{i-1})^2 ) )

    This is the scaled error "used in the M5 Forecasting Competition in
    2020" (p. 572).  ``season`` generalizes the naive lag to a seasonal
    naive one; leave it at 1 for the printed formula.
    """
    a, b = _pair(y, yhat)
    ins = _vec(insample, "insample")
    season = int(season)
    if season < 1 or len(ins) <= season:
        raise ValueError("insample must be longer than season >= 1")
    den = sum((ins[i] - ins[i - season]) ** 2 for i in range(season, len(ins)))
    den /= float(len(ins) - season)
    if den <= 0.0:
        raise ValueError("in-sample naive error is zero; RMSSE is undefined")
    num = sum((a[i] - b[i]) ** 2 for i in range(len(a))) / len(a)
    mase = (sum(abs(a[i] - b[i]) for i in range(len(a))) / len(a)) / (
        sum(abs(ins[i] - ins[i - season]) for i in range(season, len(ins)))
        / float(len(ins) - season)
    )
    return {"rmsse": math.sqrt(num / den), "scale": den, "mase": mase, "n": len(a)}


def relmae(y, yhat, benchmark):
    """Relative MAE against a benchmark forecast, ch. 19 p. 571.

    RelMAE = MAE(model) / MAE(benchmark); below 1 the model beats the
    benchmark.
    """
    a, b = _pair(y, yhat)
    c = _vec(benchmark, "benchmark")
    if len(c) != len(a):
        raise ValueError("benchmark must be the same length as y")
    mae = sum(abs(a[i] - b[i]) for i in range(len(a))) / len(a)
    base = sum(abs(a[i] - c[i]) for i in range(len(a))) / len(a)
    if base <= 0.0:
        raise ValueError("benchmark MAE is zero; RelMAE is undefined")
    return {"relmae": mae / base, "mae": mae, "benchmae": base,
            "better": bool(mae < base), "n": len(a)}


def pinball(y, qhat, q):
    """Pinball (quantile) loss, ch. 17 p. 494.

    The book points at the quantile loss for probabilistic forecasts
    ("we can use quantile loss or pinball loss", p. 494).  The
    canonical statement is TFT eq. (25), which the book's own
    architecture chapter builds on:

        QL(y, yhat, q) = q (y - yhat)_+ + (1 - q) (yhat - y)_+

    -- Lim, B., Arik, S. O., Loeff, N. and Pfister, T., "Temporal
    Fusion Transformers for Interpretable Multi-horizon Time Series
    Forecasting", arXiv:1912.09363, eq. (25).
    """
    a, b = _pair(y, qhat)
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly in (0, 1)")
    losses = [
        q * max(a[i] - b[i], 0.0) + (1.0 - q) * max(b[i] - a[i], 0.0)
        for i in range(len(a))
    ]
    cov = sum(1 for i in range(len(a)) if a[i] <= b[i]) / float(len(a))
    return {"loss": _mean(losses), "total": sum(losses), "coverage": cov,
            "q": q, "n": len(a)}


def winkler(y, lower, upper, alpha=0.1):
    """Winkler interval score, ch. 17.

    NOT LOCATED IN THE EXTRACTED TEXT: the corpus copy of Joseph and
    Tackes never prints the Winkler score, so it is taken from the
    primary source and stated here in full:

        W = (u - l)
            + (2/alpha)(l - y)  if y < l
            + (2/alpha)(y - u)  if y > u

    -- Winkler, R. L. (1972), "A Decision-Theoretic Approach to
    Interval Estimation", Journal of the American Statistical
    Association 67(337):187-191; in the form popularized by Gneiting,
    T. and Raftery, A. E. (2007), "Strictly Proper Scoring Rules,
    Prediction, and Estimation", JASA 102(477):359-378, eq. (43).
    Lower is better.
    """
    a = _vec(y, "y")
    lo = _vec(lower, "lower")
    up = _vec(upper, "upper")
    if not len(a) == len(lo) == len(up):
        raise ValueError("y, lower and upper must be the same length")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1)")
    scores = []
    inside = 0
    for i in range(len(a)):
        if up[i] < lo[i]:
            raise ValueError("upper must be at least lower at every point")
        s = up[i] - lo[i]
        if a[i] < lo[i]:
            s += (2.0 / alpha) * (lo[i] - a[i])
        elif a[i] > up[i]:
            s += (2.0 / alpha) * (a[i] - up[i])
        else:
            inside += 1
        scores.append(s)
    return {"score": _mean(scores), "total": sum(scores),
            "coverage": inside / float(len(a)),
            "meanwidth": _mean([up[i] - lo[i] for i in range(len(a))]),
            "n": len(a)}


# =====================================================================
# Transformations -- ch. 6 pp. 163-166
# =====================================================================

def boxcox(x, lam):
    """Box-Cox transformation, ch. 6 p. 164.

        w = (x^lambda - 1) / lambda   for lambda != 0
        w = log(x)                    for lambda == 0
    """
    v = _vec(x)
    lam = float(lam)
    if any(t <= 0.0 for t in v):
        raise ValueError("Box-Cox needs strictly positive values")
    if lam == 0.0:
        w = [math.log(t) for t in v]
    else:
        w = [(t ** lam - 1.0) / lam for t in v]
    return {"w": w, "lam": lam, "mean": _mean(w),
            "var": sum((t - _mean(w)) ** 2 for t in w) / len(w), "n": len(w)}


def logtrans(x, offset=0.0):
    """Log transformation with an optional offset, ch. 6 p. 163.

    The offset is the book's own remedy for series containing zeros.
    ``ratio`` reports the variance-stabilization achieved: the
    coefficient of variation before and after.
    """
    v = _vec(x)
    offset = float(offset)
    if any(t + offset <= 0.0 for t in v):
        raise ValueError("log transform needs x + offset strictly positive")
    w = [math.log(t + offset) for t in v]
    mv, mw = _mean(v), _mean(w)
    sv = math.sqrt(sum((t - mv) ** 2 for t in v) / len(v))
    sw = math.sqrt(sum((t - mw) ** 2 for t in w) / len(w))
    return {"w": w, "mean": mw, "sd": sw,
            "cvbefore": sv / abs(mv) if mv else float("nan"),
            "cvafter": sw / abs(mw) if mw else float("nan"), "n": len(w)}


def diffser(x, order=1, season=1):
    """Differencing, ch. 6 pp. 155-158.

    ``order`` successive lag-``season`` differences.  Seasonal
    differencing is the same operator at lag m.
    """
    v = _vec(x)
    order = int(order)
    season = int(season)
    if order < 1 or season < 1:
        raise ValueError("order and season must be at least 1")
    if len(v) <= order * season:
        raise ValueError("series is too short for this differencing")
    w = v
    for _ in range(order):
        w = [w[i] - w[i - season] for i in range(season, len(w))]
    mw = _mean(w)
    return {"w": w, "mean": mw,
            "var": sum((t - mw) ** 2 for t in w) / len(w),
            "dropped": len(v) - len(w), "n": len(w)}


# =====================================================================
# Feature engineering -- ch. 6 pp. 168-186
# =====================================================================

def lagfeat(x, lags):
    """Lag features, ch. 6 p. 170.

    Returns the design rows for which every requested lag exists, so
    the matrix has no missing cells.
    """
    v = _vec(x)
    lags = sorted({int(t) for t in lags})
    if not lags or lags[0] < 1:
        raise ValueError("lags must be positive integers")
    start = lags[-1]
    if len(v) <= start:
        raise ValueError("series is too short for the largest lag")
    rows = [[v[i - lg] for lg in lags] for i in range(start, len(v))]
    flat = [c for r in rows for c in r]
    return {"rows": rows, "target": v[start:], "lags": lags,
            "nrows": len(rows), "ncols": len(lags), "mean": _mean(flat)}


def rollfeat(x, window, minperiods=None):
    """Rolling-window features, ch. 6 p. 176.

    Trailing mean, standard deviation, minimum and maximum over the
    last ``window`` observations, computed only where at least
    ``minperiods`` observations are available (default: the full
    window, so no partial window ever leaks a shorter average).
    """
    v = _vec(x)
    window = int(window)
    if window < 1:
        raise ValueError("window must be at least 1")
    mp = window if minperiods is None else int(minperiods)
    if mp < 1 or mp > window:
        raise ValueError("minperiods must lie in [1, window]")
    means, sds, mins, maxs = [], [], [], []
    for i in range(len(v)):
        lo = max(0, i - window + 1)
        w = v[lo:i + 1]
        if len(w) < mp:
            continue
        m = _mean(w)
        means.append(m)
        sds.append(math.sqrt(sum((t - m) ** 2 for t in w) / len(w)))
        mins.append(min(w))
        maxs.append(max(w))
    return {"mean": means, "sd": sds, "min": mins, "max": maxs,
            "nrows": len(means), "lastmean": means[-1] if means else float("nan"),
            "meanofmeans": _mean(means) if means else float("nan")}


def fourfeat(n, period, k, start=0):
    """Fourier terms for seasonality, ch. 4 p. 61 and ch. 5 p. 95.

    Column pair j is sin(2 pi j t / m), cos(2 pi j t / m) for
    j = 1..k, evaluated at t = start .. start + n - 1.  The book calls
    these "trigonometric seasonality" (p. 95).
    """
    n = int(n)
    period = float(period)
    k = int(k)
    if n < 1 or period <= 0.0 or k < 1:
        raise ValueError("need n >= 1, period > 0 and k >= 1")
    if 2 * k > period:
        raise ValueError("k must not exceed period / 2 (Nyquist)")
    rows = []
    for i in range(n):
        t = float(int(start) + i)
        row = []
        for j in range(1, k + 1):
            ang = _TWOPI * j * t / period
            row.append(math.sin(ang))
            row.append(math.cos(ang))
        rows.append(row)
    flat = [c for r in rows for c in r]
    return {"rows": rows, "nrows": n, "ncols": 2 * k, "k": k,
            "period": period, "mean": _mean(flat),
            "sumsq": sum(c * c for c in flat)}


_MONTH_LEN = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _isleap(y):
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def _daynum(y, m, d):
    """Days since 1970-01-01, proleptic Gregorian."""
    days = 0
    if y >= 1970:
        for yy in range(1970, y):
            days += 366 if _isleap(yy) else 365
    else:
        for yy in range(y, 1970):
            days -= 366 if _isleap(yy) else 365
    for mm in range(1, m):
        days += _MONTH_LEN[mm - 1] + (1 if (mm == 2 and _isleap(y)) else 0)
    return days + d - 1


def calfeat(dates):
    """Calendar / time features, ch. 6 p. 168.

    ``dates`` is a sequence of (year, month, day) triples.  Produces
    the book's time-based features -- year, month, day, day of week,
    day of year, quarter, week-of-year, weekend flag, month-start and
    month-end flags -- plus the cyclic sine/cosine encoding of month
    and day of week that the book recommends so December sits next to
    January.

    The calendar arithmetic is proleptic Gregorian and written out
    here, so both language arms agree without either depending on a
    date library.
    """
    rows = []
    for item in dates:
        y, m, d = (int(t) for t in item)
        if not 1 <= m <= 12:
            raise ValueError("month must lie in 1..12")
        mlen = _MONTH_LEN[m - 1] + (1 if (m == 2 and _isleap(y)) else 0)
        if not 1 <= d <= mlen:
            raise ValueError("day is out of range for that month")
        dn = _daynum(y, m, d)
        dow = (dn + 4) % 7  # 1970-01-01 was a Thursday
        doy = dn - _daynum(y, 1, 1) + 1
        rows.append({
            "year": y, "month": m, "day": d, "dow": dow, "doy": doy,
            "quarter": (m - 1) // 3 + 1, "week": (doy - 1) // 7 + 1,
            "weekend": 1 if dow >= 5 else 0,
            "monthstart": 1 if d == 1 else 0,
            "monthend": 1 if d == mlen else 0,
            "monthsin": math.sin(_TWOPI * m / 12.0),
            "monthcos": math.cos(_TWOPI * m / 12.0),
            "dowsin": math.sin(_TWOPI * dow / 7.0),
            "dowcos": math.cos(_TWOPI * dow / 7.0),
        })
    return {"rows": rows, "n": len(rows),
            "nweekend": sum(r["weekend"] for r in rows),
            "meandoy": _mean([float(r["doy"]) for r in rows]),
            "meanmonthsin": _mean([r["monthsin"] for r in rows])}


def tsimpute(x, method="linear", season=1):
    """Missing-data imputation for time series, ch. 2 pp. 44-52.

    ``x`` may contain ``None`` for a gap.  ``method`` is one of the
    book's own options: ``ffill`` (last observation carried forward),
    ``bfill``, ``linear`` interpolation between the flanking
    observations, ``mean`` of the observed values, or ``seasonal``
    (the mean of the same seasonal position).  Leading or trailing
    gaps that a method cannot reach fall back to the series mean, so
    the output never contains a hole.
    """
    raw = list(x)
    if not raw:
        raise ValueError("x must be non-empty")
    obs = [(i, float(v)) for i, v in enumerate(raw) if v is not None]
    if not obs:
        raise ValueError("x contains no observed values")
    known = dict(obs)
    gm = sum(v for _, v in obs) / len(obs)
    n = len(raw)
    season = int(season)
    if season < 1:
        raise ValueError("season must be at least 1")
    idx = [i for i, _ in obs]
    out = []
    for i in range(n):
        if i in known:
            out.append(known[i])
            continue
        prev = [j for j in idx if j < i]
        nxt = [j for j in idx if j > i]
        if method == "ffill":
            out.append(known[prev[-1]] if prev else gm)
        elif method == "bfill":
            out.append(known[nxt[0]] if nxt else gm)
        elif method == "mean":
            out.append(gm)
        elif method == "seasonal":
            same = [known[j] for j in idx if (j - i) % season == 0]
            out.append(sum(same) / len(same) if same else gm)
        elif method == "linear":
            if prev and nxt:
                a, b = prev[-1], nxt[0]
                w = (i - a) / float(b - a)
                out.append(known[a] + w * (known[b] - known[a]))
            elif prev:
                out.append(known[prev[-1]])
            elif nxt:
                out.append(known[nxt[0]])
            else:
                out.append(gm)
        else:
            raise ValueError("unknown method %r" % (method,))
    return {"x": out, "nmissing": n - len(obs), "n": n,
            "mean": _mean(out), "method": method,
            "imputedmean": _mean([out[i] for i in range(n) if i not in known])
            if n - len(obs) else float("nan")}


# =====================================================================
# Diagnostics -- ch. 3 pp. 61-68 and ch. 6 p. 149
# =====================================================================

def autocorf(x, maxlag=20):
    """Autocorrelation function, ch. 3.

    The standard biased (divide-by-n) estimator, which is what the
    book's ACF plots use:

        r_k = sum_{t=k+1..n} (x_t - xbar)(x_{t-k} - xbar)
              / sum_{t=1..n} (x_t - xbar)^2

    ``ci`` is the +/- 1.96/sqrt(n) band drawn on those plots.
    """
    v = _vec(x)
    n = len(v)
    maxlag = int(maxlag)
    if maxlag < 1 or maxlag >= n:
        raise ValueError("maxlag must lie in [1, n - 1]")
    m = _mean(v)
    den = sum((t - m) ** 2 for t in v)
    if den <= 0.0:
        raise ValueError("series is constant; the ACF is undefined")
    r = [sum((v[t] - m) * (v[t - k] - m) for t in range(k, n)) / den
         for k in range(0, maxlag + 1)]
    ci = 1.96 / math.sqrt(n)
    return {"acf": r, "ci": ci, "maxlag": maxlag, "n": n,
            "r1": r[1], "nsignif": sum(1 for k in range(1, maxlag + 1)
                                       if abs(r[k]) > ci)}


def pacfts(x, maxlag=20):
    """Partial autocorrelation by the Durbin-Levinson recursion, ch. 3.

    Fixed recursion depth, no tolerance test, so both arms take
    identical steps.
    """
    v = _vec(x)
    n = len(v)
    maxlag = int(maxlag)
    if maxlag < 1 or maxlag >= n:
        raise ValueError("maxlag must lie in [1, n - 1]")
    r = autocorf(v, maxlag)["acf"]
    phi = [[0.0] * (maxlag + 1) for _ in range(maxlag + 1)]
    pacf = [1.0]
    if r[1] == 1.0:
        raise ValueError("series is perfectly autocorrelated at lag 1")
    phi[1][1] = r[1]
    pacf.append(r[1])
    for k in range(2, maxlag + 1):
        num = r[k] - sum(phi[k - 1][j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1][j] * r[j] for j in range(1, k))
        if abs(den) < 1e-300:
            raise ValueError("Durbin-Levinson recursion broke down at lag %d" % k)
        phi[k][k] = num / den
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
        pacf.append(phi[k][k])
    ci = 1.96 / math.sqrt(n)
    return {"pacf": pacf, "ci": ci, "maxlag": maxlag, "n": n,
            "p1": pacf[1], "nsignif": sum(1 for k in range(1, maxlag + 1)
                                          if abs(pacf[k]) > ci)}


# Dickey-Fuller critical values, MacKinnon (1991) response-surface
# constants for the constant-only ("c") regression, the case the book
# uses on p. 149. Reported so the caller can compare without a table.
_DF_C = {0.01: (-3.43035, -6.5393, -16.786), 0.05: (-2.86154, -2.8903, -4.234),
         0.10: (-2.56677, -1.5384, -2.809)}


def adfur(x, lags=1):
    """Augmented Dickey-Fuller unit-root test, ch. 6 p. 149.

    Regresses ``diff(x)_t`` on a constant, ``x_{t-1}`` and ``lags``
    lagged differences; the statistic is the t-ratio on ``x_{t-1}``.
    The null is a unit root, so a statistic BELOW the critical value
    rejects non-stationarity -- which is the direction the book uses on
    p. 149.

    Critical values are MacKinnon's (1991) response surface for the
    constant-only regression, ``tau = b0 + b1/n + b2/n^2``; they are
    returned rather than a p-value, because interpolating a p-value
    would need a table the book does not print.
    """
    v = _vec(x)
    lags = int(lags)
    if lags < 0:
        raise ValueError("lags must be non-negative")
    d = [v[i] - v[i - 1] for i in range(1, len(v))]
    start = lags
    n = len(d) - start
    if n <= lags + 3:
        raise ValueError("series is too short for %d augmenting lags" % lags)
    rows, y = [], []
    for i in range(start, len(d)):
        row = [1.0, v[i]]
        for j in range(1, lags + 1):
            row.append(d[i - j])
        rows.append(row)
        y.append(d[i])
    beta = _ols(rows, y)
    p = len(beta)
    resid = [y[i] - sum(rows[i][k] * beta[k] for k in range(p))
             for i in range(n)]
    dof = n - p
    if dof < 1:
        raise ValueError("not enough degrees of freedom")
    s2 = sum(t * t for t in resid) / dof
    xtx = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(p)]
           for a in range(p)]
    for a in range(p):
        xtx[a][a] += 1e-12
    e1 = [1.0 if k == 1 else 0.0 for k in range(p)]
    se = math.sqrt(s2 * _solve(xtx, e1)[1])
    stat = beta[1] / se
    crit = {}
    for lvl, (b0, b1, b2) in _DF_C.items():
        crit[lvl] = b0 + b1 / n + b2 / (n * n)
    return {"stat": stat, "gamma": beta[1], "se": se, "lags": lags,
            "n": n, "crit1": crit[0.01], "crit5": crit[0.05],
            "crit10": crit[0.10],
            "stationary5": bool(stat < crit[0.05])}


def stldecomp(x, period, robust=False, iters=2):
    """Seasonal-trend decomposition, ch. 3 p. 64.

    The book's STL uses LOESS smoothers.  This routine uses the
    classical moving-average form of the same three-part model --
    centred moving-average trend, seasonal means of the detrended
    series, remainder -- iterated ``iters`` times.  That substitution
    is OURS and is stated here rather than passed off as STL: the
    LOESS smoother has bandwidth and robustness-iteration choices whose
    defaults differ between implementations, and a decomposition whose
    numbers depend on which library you call cannot be checked across
    two languages.  ``robust`` switches the seasonal aggregate from the
    mean to the median, which is the robustness knob the book
    describes.

    The additive model is x = trend + seasonal + remainder, and the
    seasonal component is centred to sum to zero over one period, as
    STL also does.
    """
    v = _vec(x)
    period = int(period)
    iters = int(iters)
    if period < 2 or len(v) < 2 * period:
        raise ValueError("need period >= 2 and at least two full periods")
    if iters < 1:
        raise ValueError("iters must be at least 1")
    n = len(v)
    half = period // 2
    seasonal = [0.0] * n
    trend = [0.0] * n
    for _ in range(iters):
        deseas = [v[i] - seasonal[i] for i in range(n)]
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            w = deseas[lo:hi]
            trend[i] = sum(w) / len(w)
        detr = [v[i] - trend[i] for i in range(n)]
        agg = []
        for s in range(period):
            grp = sorted(detr[i] for i in range(s, n, period))
            if robust:
                mid = len(grp) // 2
                agg.append(grp[mid] if len(grp) % 2
                           else 0.5 * (grp[mid - 1] + grp[mid]))
            else:
                agg.append(sum(grp) / len(grp))
        off = sum(agg) / period
        agg = [t - off for t in agg]
        seasonal = [agg[i % period] for i in range(n)]
    remainder = [v[i] - trend[i] - seasonal[i] for i in range(n)]
    vr = sum(t * t for t in remainder) / n
    vv = sum((t - _mean(v)) ** 2 for t in v) / n
    return {"trend": trend, "seasonal": seasonal, "remainder": remainder,
            "period": period, "n": n,
            "seasonalstrength": max(0.0, 1.0 - vr / max(vv, 1e-300)),
            "remaindervar": vr,
            "seasonalrange": max(seasonal) - min(seasonal)}


# =====================================================================
# Multi-step strategies -- ch. 18 pp. 545-555
# =====================================================================

def tsregmat(x, lags, horizon=1):
    """Time series as a regression problem, ch. 5 p. 118.

    Builds the supervised design: one row per usable time index, the
    requested lags as columns, and the value ``horizon`` steps ahead as
    the target.  Every multi-step strategy below consumes this.
    """
    v = _vec(x)
    lags = sorted({int(t) for t in lags})
    horizon = int(horizon)
    if not lags or lags[0] < 1 or horizon < 1:
        raise ValueError("lags must be positive and horizon at least 1")
    start = lags[-1]
    rows, y = [], []
    for i in range(start, len(v) - horizon + 1):
        rows.append([v[i - lg] for lg in lags])
        y.append(v[i + horizon - 1])
    if not rows:
        raise ValueError("series is too short for these lags and horizon")
    return {"rows": rows, "y": y, "lags": lags, "horizon": horizon,
            "nrows": len(rows), "ncols": len(lags),
            "ymean": _mean(y), "xmean": _mean([c for r in rows for c in r])}


def _fit_predict(rows, y, newrow):
    design = [[1.0] + r for r in rows]
    beta = _ols(design, y)
    return sum(([1.0] + newrow)[k] * beta[k] for k in range(len(beta))), beta


def recmulti(x, lags, horizon):
    """Recursive multi-step forecasting, ch. 18 p. 546.

    One model, trained for a single step, applied repeatedly with its
    own forecasts fed back in as lags.  The base learner is ordinary
    least squares on the lag design, so the strategy -- which is what
    the book is teaching -- is what is being demonstrated, and nothing
    is fitted at random.
    """
    v = _vec(x)
    lags = sorted({int(t) for t in lags})
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    tr = tsregmat(v, lags, 1)
    hist = list(v)
    preds = []
    for _ in range(horizon):
        newrow = [hist[len(hist) - lg] for lg in lags]
        p, _b = _fit_predict(tr["rows"], tr["y"], newrow)
        preds.append(p)
        hist.append(p)
    return {"forecast": preds, "horizon": horizon, "nmodels": 1,
            "ntrain": tr["nrows"], "first": preds[0], "last": preds[-1],
            "mean": _mean(preds)}


def dirmulti(x, lags, horizon):
    """Direct multi-step forecasting, ch. 18 p. 548.

    One model PER horizon, each trained to predict h steps ahead
    directly from the same observed lags -- so no forecast is ever fed
    back in, and errors cannot compound the way they do in the
    recursive strategy.
    """
    v = _vec(x)
    lags = sorted({int(t) for t in lags})
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    newrow = [v[len(v) - lg] for lg in lags]
    preds = []
    for h in range(1, horizon + 1):
        tr = tsregmat(v, lags, h)
        p, _b = _fit_predict(tr["rows"], tr["y"], newrow)
        preds.append(p)
    return {"forecast": preds, "horizon": horizon, "nmodels": horizon,
            "first": preds[0], "last": preds[-1], "mean": _mean(preds)}


def dirrec(x, lags, horizon):
    """DirRec strategy, ch. 18 p. 551.

    The hybrid the book names: like Direct, a separate model per
    horizon; like Recursive, each successive model may also use the
    forecasts already produced, so the input space GROWS by one column
    at every step.
    """
    v = _vec(x)
    lags = sorted({int(t) for t in lags})
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    preds = []
    hist = list(v)
    ncols = []
    for h in range(1, horizon + 1):
        extra = len(preds)
        base = tsregmat(v, lags, h)
        rows = []
        ys = []
        for i, r in enumerate(base["rows"]):
            # the extra columns are the h-1 values immediately after
            # the lag window, i.e. what the earlier models forecast
            pos = lags[-1] + i
            if pos + extra >= len(v):
                continue
            rows.append(list(r) + [v[pos + j] for j in range(extra)])
            ys.append(base["y"][i])
        if not rows:
            raise ValueError("series is too short for the DirRec expansion")
        newrow = [hist[len(v) - lg] for lg in lags] + list(preds)
        p, _b = _fit_predict(rows, ys, newrow)
        preds.append(p)
        ncols.append(len(newrow))
    return {"forecast": preds, "horizon": horizon, "nmodels": horizon,
            "ncolsfirst": ncols[0], "ncolslast": ncols[-1],
            "first": preds[0], "last": preds[-1], "mean": _mean(preds)}


def seasnaive(x, season, horizon):
    """Seasonal naive baseline, ch. 8 p. 219.

    Each forecast repeats the observation from the same point in the
    previous season -- the benchmark the scaled metrics divide by.
    """
    v = _vec(x)
    season = int(season)
    horizon = int(horizon)
    if season < 1 or horizon < 1:
        raise ValueError("season and horizon must be at least 1")
    if len(v) < season:
        raise ValueError("series is shorter than one season")
    preds = [v[len(v) - season + ((h) % season)] for h in range(horizon)]
    return {"forecast": preds, "season": season, "horizon": horizon,
            "first": preds[0], "last": preds[-1], "mean": _mean(preds)}


# =====================================================================
# Validation -- ch. 5 pp. 126-133
# =====================================================================

def slidecv(n, trainsize, testsize, step=None):
    """Sliding-window cross-validation, ch. 5 p. 128.

    A fixed-length training window slides forward, so old data drops
    out.  Returns the fold boundaries as half-open [start, end) index
    pairs; the caller fits whatever it likes on them.
    """
    n, trainsize, testsize = int(n), int(trainsize), int(testsize)
    step = testsize if step is None else int(step)
    if min(n, trainsize, testsize, step) < 1:
        raise ValueError("all arguments must be positive")
    folds = []
    s = 0
    while s + trainsize + testsize <= n:
        folds.append((s, s + trainsize, s + trainsize, s + trainsize + testsize))
        s += step
    if not folds:
        raise ValueError("n is too small for this window layout")
    return {"folds": folds, "nfolds": len(folds), "trainsize": trainsize,
            "testsize": testsize, "step": step,
            "firsttest": folds[0][2], "lasttest": folds[-1][3]}


def expandcv(n, initial, testsize, step=None):
    """Expanding-window cross-validation, ch. 5 p. 130.

    The training window GROWS: every fold starts at index 0, so no
    history is ever discarded.
    """
    n, initial, testsize = int(n), int(initial), int(testsize)
    step = testsize if step is None else int(step)
    if min(n, initial, testsize, step) < 1:
        raise ValueError("all arguments must be positive")
    folds = []
    end = initial
    while end + testsize <= n:
        folds.append((0, end, end, end + testsize))
        end += step
    if not folds:
        raise ValueError("n is too small for this window layout")
    return {"folds": folds, "nfolds": len(folds), "initial": initial,
            "testsize": testsize, "step": step,
            "firsttrainend": folds[0][1], "lasttrainend": folds[-1][1]}


def walkfwd(y, yhat, initial, testsize, step=None):
    """Walk-forward validation, ch. 5 p. 126.

    Scores an already-produced forecast series fold by fold on an
    expanding-window layout, and reports the fold RMSEs plus their mean
    and spread -- which is the number the book actually reads off a
    walk-forward run.
    """
    a, b = _pair(y, yhat)
    lay = expandcv(len(a), initial, testsize, step)
    scores = []
    for (_ts, _te, s, e) in lay["folds"]:
        errs = [(a[i] - b[i]) ** 2 for i in range(s, e)]
        scores.append(math.sqrt(sum(errs) / len(errs)))
    m = _mean(scores)
    return {"scores": scores, "nfolds": len(scores), "rmse": m,
            "sd": math.sqrt(sum((t - m) ** 2 for t in scores) / len(scores)),
            "best": min(scores), "worst": max(scores)}


# =====================================================================
# Probabilistic forecasting -- ch. 17 pp. 494-520
# =====================================================================

def quantreg(x, y, q, iters=25):
    """Linear quantile regression, ch. 17 p. 500.

    Fitted by iteratively reweighted least squares on the pinball loss
    with a fixed iteration count and a fixed smoothing floor -- no
    convergence test, so both arms take identical steps.  ``x`` is a
    list of feature rows; an intercept is added.
    """
    rows = [[1.0] + [float(t) for t in r] for r in x]
    yv = _vec(y, "y")
    if len(rows) != len(yv):
        raise ValueError("x and y must have the same number of rows")
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly in (0, 1)")
    iters = int(iters)
    if iters < 1:
        raise ValueError("iters must be at least 1")
    p = len(rows[0])
    n = len(rows)
    beta = _ols(rows, yv)
    eps = 1e-6
    for _ in range(iters):
        w = []
        for i in range(n):
            r = yv[i] - sum(rows[i][k] * beta[k] for k in range(p))
            w.append((q if r > 0.0 else (1.0 - q)) / max(abs(r), eps))
        xtx = [[sum(w[i] * rows[i][a] * rows[i][b] for i in range(n))
                for b in range(p)] for a in range(p)]
        xty = [sum(w[i] * rows[i][a] * yv[i] for i in range(n)) for a in range(p)]
        for a in range(p):
            xtx[a][a] += 1e-10
        beta = _solve(xtx, xty)
    fit = [sum(rows[i][k] * beta[k] for k in range(p)) for i in range(n)]
    loss = pinball(yv, fit, q)["loss"]
    return {"beta": beta, "fitted": fit, "loss": loss, "q": q,
            "intercept": beta[0], "n": n, "p": p}


def cqr(callo, calhi, caly, lo, hi, alpha=0.1):
    """Conformalized quantile regression, ch. 17 pp. 514-515.

    The book's own non-conformity score, quoted from p. 514:

        s(x, y) = max{ yhat_t^{alpha/2} - y , y - yhat_t^{1-(alpha/2)} }

    The conformal quantile of those calibration scores is then added to
    both ends of the test interval.  The rank used is the standard
    finite-sample one, ceil((n+1)(1-alpha))/n, which is what delivers
    the coverage guarantee the section is about.

    -- the method is Romano, Y., Patterson, E. and Candes, E. (2019),
    "Conformalized Quantile Regression", NeurIPS 32 (arXiv:1905.03222),
    which the book cites as its Reference 11.
    """
    cl = _vec(callo, "callo")
    ch = _vec(calhi, "calhi")
    cy = _vec(caly, "caly")
    if not len(cl) == len(ch) == len(cy):
        raise ValueError("calibration arrays must be the same length")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1)")
    scores = sorted(max(cl[i] - cy[i], cy[i] - ch[i]) for i in range(len(cy)))
    n = len(scores)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    qhat = scores[min(k, n) - 1] if k >= 1 else scores[0]
    lo = _vec(lo, "lo")
    hi = _vec(hi, "hi")
    if len(lo) != len(hi):
        raise ValueError("lo and hi must be the same length")
    newlo = [t - qhat for t in lo]
    newhi = [t + qhat for t in hi]
    return {"qhat": qhat, "lower": newlo, "upper": newhi, "k": k, "n": n,
            "meanwidth": _mean([newhi[i] - newlo[i] for i in range(len(lo))]),
            "widening": 2.0 * qhat}


def aci(inside, alpha=0.1, gamma=0.01):
    """Adaptive conformal inference, ch. 17 p. 519.

    The book's own online update, quoted from p. 519:

        err_t = 1 if Y_t is outside Chat(alpha_t), else 0
        alpha_{t+1} = alpha_t + gamma (alpha - err_t)

    with alpha_1 = alpha.  ``inside`` is the sequence of coverage
    outcomes (True when the observation fell inside the interval), so
    the routine is a pure recursion over data the caller supplies.

    -- the method is Gibbs, I. and Candes, E. (2021), "Adaptive
    Conformal Inference Under Distribution Shift", NeurIPS 34
    (arXiv:2106.00170), which the book cites as its Reference 13.
    """
    seq = [bool(t) for t in inside]
    if not seq:
        raise ValueError("inside must be non-empty")
    alpha = float(alpha)
    gamma = float(gamma)
    if not 0.0 < alpha < 1.0 or gamma <= 0.0:
        raise ValueError("alpha must lie in (0, 1) and gamma be positive")
    at = alpha
    path = [at]
    nerr = 0
    for ok in seq:
        err = 0.0 if ok else 1.0
        nerr += int(err)
        at = at + gamma * (alpha - err)
        path.append(at)
    return {"alpha": path, "final": at, "empirical": nerr / float(len(seq)),
            "target": alpha, "gamma": gamma, "n": len(seq),
            "minalpha": min(path), "maxalpha": max(path)}


# =====================================================================
# Deep architectures -- ch. 14-16, ch. 20
#
# The book NAMES each of these and cites its paper; it does not print
# their equations. Every equation below was therefore taken from the
# paper itself and is quoted in the docstring with the paper's own
# equation number. Every weight is CALLER-SUPPLIED, so no layer
# initializes anything at random and the R mirror matches exactly.
# =====================================================================

def _matvec(w, v):
    if any(len(r) != len(v) for r in w):
        raise ValueError("weight row length must match the input length")
    return [sum(r[j] * v[j] for j in range(len(v))) for r in w]


def _addv(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    return [a[i] + b[i] for i in range(len(a))]


def _softmaxv(v):
    m = max(v)
    ex = [math.exp(t - m) for t in v]
    s = sum(ex)
    return [t / s for t in ex]


def _layernorm(v, eps=1e-5):
    m = _mean(v)
    var = sum((t - m) ** 2 for t in v) / len(v)
    d = math.sqrt(var + eps)
    return [(t - m) / d for t in v]


def _elu(t):
    return t if t > 0.0 else math.expm1(t)


def _relu(t):
    return t if t > 0.0 else 0.0


def _sigmoid(t):
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


def _maxpool(v, k):
    k = int(k)
    if k < 1:
        raise ValueError("pool kernel must be at least 1")
    if k == 1:
        return list(v)
    out = []
    for s in range(0, len(v), k):
        w = v[s:s + k]
        if w:
            out.append(max(w))
    return out


def _interp(theta, length):
    """Linear interpolation g(tau, theta), N-HiTS eq. (4)."""
    n = len(theta)
    if n < 1 or length < 1:
        raise ValueError("theta and length must be non-empty")
    if n == 1:
        return [theta[0]] * length
    out = []
    for i in range(length):
        pos = (i * (n - 1)) / float(length - 1) if length > 1 else 0.0
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        w = pos - lo
        out.append(theta[lo] + (theta[hi] - theta[lo]) * w)
    return out


def seriesdecomp(x, kernel):
    """Autoformer series decomposition block, eq. (1).

    Quoted from the paper:
        "X_t = AvgPool(Padding(X)), X_s = X - X_t"

    -- Wu, H., Xu, J., Wang, J. and Long, M., "Autoformer:
    Decomposition Transformers with Auto-Correlation for Long-Term
    Series Forecasting", NeurIPS 2021 (arXiv:2106.13008), eq. (1).
    The padding is the paper's edge replication, which keeps the trend
    the same length as the input.
    """
    v = _vec(x)
    kernel = int(kernel)
    if kernel < 1:
        raise ValueError("kernel must be at least 1")
    half = kernel // 2
    pad = [v[0]] * half + v + [v[-1]] * (kernel - 1 - half)
    trend = [sum(pad[i:i + kernel]) / kernel for i in range(len(v))]
    seas = [v[i] - trend[i] for i in range(len(v))]
    return {"trend": trend, "seasonal": seas, "kernel": kernel, "n": len(v),
            "trendmean": _mean(trend), "seasmean": _mean(seas),
            "seasrange": max(seas) - min(seas)}


def autoform(q, k, v, kernel=3, c=1.0):
    """Autoformer decomposition plus Auto-Correlation, eqs. (1), (5), (6).

    Quoted from the paper:
        (5)  "R_XX(tau) = lim_{L->inf} (1/L) sum_{t=1..L} X_t X_{t-tau}"
        (6)  "tau_1,...,tau_k = arg Topk_{tau in {1..L}}(R_{Q,K}(tau))"
             with "k = floor(c x log L)"
             "Rhat_{Q,K}(tau_1),...,Rhat_{Q,K}(tau_k)
                  = SoftMax(R_{Q,K}(tau_1),...,R_{Q,K}(tau_k))"
             "Auto-Correlation(Q,K,V)
                  = sum_{i=1..k} Roll(V, tau_i) Rhat_{Q,K}(tau_i)"

    -- Wu, H., Xu, J., Wang, J. and Long, M., "Autoformer", NeurIPS
    2021 (arXiv:2106.13008).  ``Roll`` is the circular shift the paper
    uses to align sub-series; ties in the Topk are broken by the
    smaller lag so the selection is deterministic.
    """
    qv, kv = _pair(q, k)
    vv = _vec(v, "v")
    if len(vv) != len(qv):
        raise ValueError("q, k and v must be the same length")
    L = len(qv)
    dec = seriesdecomp(qv, kernel)
    r = [sum(qv[t] * kv[(t - tau) % L] for t in range(L)) / L for tau in range(L)]
    kk = int(math.floor(c * math.log(L))) if L > 1 else 1
    kk = max(1, min(kk, L - 1))
    cand = list(range(1, L))
    order = sorted(cand, key=lambda tau: (-r[tau], tau))[:kk]
    taus = sorted(order)
    weights = _softmaxv([r[t] for t in taus])
    out = [0.0] * L
    for w, tau in zip(weights, taus):
        for t in range(L):
            out[t] += w * vv[(t - tau) % L]
    return {"out": out, "taus": taus, "weights": weights, "k": kk, "L": L,
            "r1": r[1] if L > 1 else float("nan"),
            "outmean": _mean(out), "outmax": max(out),
            "trendmean": dec["trendmean"], "seasrange": dec["seasrange"]}


def patchts(x, patchlen, stride, eps=1e-5):
    """PatchTST patching with reversible instance normalization.

    Quoted from the paper: "the patching process will generate the a
    sequence of patches where N is the number of patches,
    N = floor((L - P)/S) + 2", with "S repeated numbers of the last
    value" padded before patching; each series is normalized to "zero
    mean and unit standard deviation" and the statistics restored on
    output; and channel-independence means a multivariate series is
    "split to M univariate series ... each of them is fed
    independently into the Transformer backbone".

    -- Nie, Y., Nguyen, N. H., Sinthong, P. and Kalagnanam, J., "A Time
    Series is Worth 64 Words: Long-term Forecasting with Transformers",
    ICLR 2023 (arXiv:2211.14730), sec. 3.1.

    ``x`` may be a single series or a list of channels; each channel is
    handled on its own, which IS the channel-independence claim.
    """
    chans = x if (x and isinstance(x[0], (list, tuple))) else [x]
    chans = [_vec(c, "channel") for c in chans]
    P = int(patchlen)
    S = int(stride)
    if P < 1 or S < 1:
        raise ValueError("patchlen and stride must be positive")
    L = len(chans[0])
    if any(len(c) != L for c in chans):
        raise ValueError("all channels must be the same length")
    if L < P:
        raise ValueError("series is shorter than one patch")
    N = (L - P) // S + 2
    allpatches = []
    stats = []
    for c in chans:
        m = _mean(c)
        sd = math.sqrt(sum((t - m) ** 2 for t in c) / len(c) + eps)
        z = [(t - m) / sd for t in c]
        padded = z + [z[-1]] * S
        patches = []
        for i in range(N):
            s = i * S
            if s + P > len(padded):
                break
            patches.append(padded[s:s + P])
        allpatches.append(patches)
        stats.append((m, sd))
    flat = [t for ch in allpatches for p in ch for t in p]
    return {"patches": allpatches, "npatches": len(allpatches[0]),
            "n": N, "patchlen": P, "stride": S, "nchannels": len(chans),
            "mean": stats[0][0], "sd": stats[0][1],
            "patchmean": _mean(flat),
            "patchsumsq": sum(t * t for t in flat)}


def nhitsnet(y, horizon, kernels, ratios, wf, wb):
    """N-HiTS multi-rate sampling with hierarchical interpolation.

    Quoted from the paper:
        (1)  "y^(p)_{t-L:t,l} = MaxPool(y_{t-L:t,l}, k_l)"
        (2)  "h_l = MLP_l(y^(p)_{t-L:t,l}); theta^f_l = LINEAR^f(h_l);
              theta^b_l = LINEAR^b(h_l)"
        (3)  "yhat_{tau,l} = g(tau, theta^f_l) ... ytilde_{tau,l}
              = g(tau, theta^b_l)"   with "|theta^f_l| = ceil(r_l H)"
        (4)  "g(tau, theta) = theta[t1]
              + ((theta[t2] - theta[t1])/(t2 - t1))(tau - t1)"
        doubly residual stacking:
             "yhat_{t+1:t+H} = sum_l yhat_{t+1:t+H,l};
              y_{t-L:t,l+1} = y_{t-L:t,l} - ytilde_{t-L:t,l}"

    -- Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F.,
    Mergenthaler-Canseco, M. and Dubrawski, A., "N-HiTS: Neural
    Hierarchical Interpolation for Time Series Forecasting", AAAI 2023
    (arXiv:2201.12886).

    ``wf[l]`` and ``wb[l]`` stand in for the paper's MLP_l followed by
    LINEAR^f / LINEAR^b: a single caller-supplied linear map from the
    pooled window to the coefficients.  That collapse is stated here
    rather than hidden -- it is what makes the block deterministic
    without a trained network -- and the expressivity ratio ``r_l``
    still governs how many coefficients each block gets, which is the
    hierarchical part the paper is actually about.
    """
    v = _vec(y, "y")
    H = int(horizon)
    if H < 1:
        raise ValueError("horizon must be at least 1")
    ks = [int(t) for t in kernels]
    rs = [float(t) for t in ratios]
    if not ks or len(ks) != len(rs) or len(ks) != len(wf) or len(ks) != len(wb):
        raise ValueError("kernels, ratios, wf and wb must line up")
    resid = list(v)
    fc = [0.0] * H
    sizes = []
    for l in range(len(ks)):
        pooled = _maxpool(resid, ks[l])
        need = int(math.ceil(rs[l] * H))
        if need < 1:
            raise ValueError("ratio %r gives no coefficients" % (rs[l],))
        thf = _matvec(wf[l], pooled)
        thb = _matvec(wb[l], pooled)
        if len(thf) != need:
            raise ValueError(
                "wf[%d] must produce ceil(r_l H) = %d coefficients" % (l, need))
        sizes.append(need)
        f = _interp(thf, H)
        b = _interp(thb, len(resid))
        fc = [fc[i] + f[i] for i in range(H)]
        resid = [resid[i] - b[i] for i in range(len(resid))]
    return {"forecast": fc, "residual": resid, "nblocks": len(ks),
            "sizes": sizes, "first": fc[0], "last": fc[-1],
            "mean": _mean(fc), "residnorm": math.sqrt(sum(t * t for t in resid))}


def _glu(gamma, w4, b4, w5, b5):
    """TFT eq. (5): GLU(gamma) = sigmoid(W4 gamma + b4) * (W5 gamma + b5)."""
    a = _addv(_matvec(w4, gamma), b4)
    b = _addv(_matvec(w5, gamma), b5)
    return [_sigmoid(a[i]) * b[i] for i in range(len(a))]


def tftnet(a, w1, b1, w2, b2, w4, b4, w5, b5, wsel, bsel, wq, bq,
           c=None, wc=None, y=None, q=0.5):
    """Temporal Fusion Transformer gating and variable selection.

    Quoted from the paper:
        (2)-(4) "GRN_omega(a, c) = LayerNorm(a + GLU_omega(eta_1))",
                with eta_1 a linear map of eta_2 and
                eta_2 = ELU(W_2 a + W_3 c + b_2)
        (5)  "GLU_omega(gamma)
                 = sigma(W_4,omega gamma + b_4,omega)
                   * (W_5,omega gamma + b_5,omega)"
        (6)  "v_chi_t = Softmax(GRN_v_chi(Xi_t, c_s))"
        (23) "yhat(q, t, tau) = W_q psitilde(t, tau) + b_q"
        (25) "QL(y, yhat, q) = q(y - yhat)_+ + (1 - q)(yhat - y)_+"

    -- Lim, B., Arik, S. O., Loeff, N. and Pfister, T., "Temporal
    Fusion Transformers for Interpretable Multi-horizon Time Series
    Forecasting", International Journal of Forecasting 37(4):1748-1764
    (arXiv:1912.09363).

    All weights are caller-supplied.  ``c`` is the optional static
    context of eq. (3), ``wc`` its projection; omit both for the
    context-free form.  Supplying ``y`` also evaluates eq. (25).
    """
    av = _vec(a, "a")
    eta2 = _addv(_matvec(w2, av), b2)
    if c is not None:
        if wc is None:
            raise ValueError("wc is required when a static context c is given")
        eta2 = _addv(eta2, _matvec(wc, _vec(c, "c")))
    eta2 = [_elu(t) for t in eta2]
    eta1 = _addv(_matvec(w1, eta2), b1)
    gated = _glu(eta1, w4, b4, w5, b5)
    grn = _layernorm(_addv(av, gated))
    sel = _softmaxv(_addv(_matvec(wsel, grn), bsel))
    yhat = _addv(_matvec(wq, grn), bq)
    out = {"grn": grn, "gate": gated, "weights": sel, "yhat": yhat,
           "topvar": max(range(len(sel)), key=lambda i: sel[i]),
           "maxweight": max(sel), "entropy": -sum(t * math.log(t)
                                                  for t in sel if t > 0.0),
           "grnnorm": math.sqrt(sum(t * t for t in grn)),
           "yhatmean": _mean(yhat)}
    if y is not None:
        out["ql"] = pinball(_vec(y, "y"), yhat, q)["loss"]
    return out


def _residblock(x, w1, b1, w2, b2, wskip, dropout=0.0):
    """TiDE residual block: MLP with ReLU, linear skip, LayerNorm."""
    h = [_relu(t) for t in _addv(_matvec(w1, x), b1)]
    out = _addv(_matvec(w2, h), b2)
    skip = _matvec(wskip, x)
    return _layernorm(_addv(out, skip))


def tide(y, feats, fproj, enc, dec, tdec, wglobal, horizon):
    """TiDE dense encoder-decoder.

    Quoted from the paper:
        (3)  "xtilde_{i,t} = ResidualBlock(x_{i,t})"
        (4)  "e^(i) = Encoder(y^i_{1:L}; xtilde^i_{1:L+H}; a^(i))"
        (5)  "g^(i) = Decoder(e^(i)) in R^{p.H}"
        (6)  "D^(i) = Reshape(g^(i)) in R^{p x H}"
             "yhat^i_{L+t} = TemporalDecoder(d_{i,t}; xtilde^i_{L+t})"

    -- Das, A., Kong, W., Leach, A., Mathur, S., Sen, R. and Yu, R.,
    "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR
    2023 (arXiv:2304.08424).  The paper states the residual block and
    the global linear residual connection in prose rather than as
    numbered equations, so those two are implemented from the prose and
    said to be so here.

    ``fproj``, ``enc``, ``dec`` and ``tdec`` are each a caller-supplied
    (w1, b1, w2, b2, wskip) residual block; ``wglobal`` is the global
    linear map from the lookback straight to the horizon, which the
    paper adds to the output.  ``tdec`` must produce one value per
    horizon step.
    """
    v = _vec(y, "y")
    H = int(horizon)
    if H < 1:
        raise ValueError("horizon must be at least 1")
    proj = [_residblock(_vec(f, "feature"), *fproj) for f in (feats or [])]
    flat = list(v)
    for pr in proj:
        flat = flat + list(pr)
    e = _residblock(flat, *enc)
    g = _residblock(e, *dec)
    if len(g) % H:
        raise ValueError("decoder output length must be a multiple of horizon")
    p = len(g) // H
    d = [[g[t * p + j] for j in range(p)] for t in range(H)]
    temporal = []
    for t in range(H):
        out1 = _residblock(d[t], *tdec)
        if len(out1) != 1:
            raise ValueError("tdec must produce exactly one value per step")
        temporal.append(out1[0])
    glob = _matvec(wglobal, v)
    if len(glob) != H:
        raise ValueError("wglobal must map the lookback to the horizon")
    out = [temporal[t] + glob[t] for t in range(H)]
    return {"forecast": out, "temporal": temporal, "global": glob,
            "horizon": H, "p": p, "encdim": len(e), "nfeat": len(proj),
            "first": out[0], "last": out[-1], "mean": _mean(out)}


def tsmixer(x, wtime, btime, wfeat, bfeat, wproj, bproj, horizon):
    """TSMixer time-mixing and feature-mixing, all-MLP.

    Quoted from the paper:
        (4)  "TP_{L->T}(X)_{*,i} = W_1 X_{*,i} + b_1, for all i = 1..C"
        (5)  "TM(X)_{*,i} = Norm(X_{*,i} + Drop(sigma(TP_{L->L}(X)_{*,i})))"

    -- Chen, S.-A., Li, C.-L., Yoder, N. C., Arik, S. O. and Pfister,
    T., "TSMixer: An All-MLP Architecture for Time Series Forecasting",
    TMLR 2023 (arXiv:2303.06053), Appendix B.3.1.  The paper describes
    feature mixing and the 2D normalization in prose rather than as
    numbered equations; both are implemented from that prose and said
    to be so here.  Dropout is omitted because it is a training-time
    stochastic operation and this routine is evaluation-time and
    deterministic.

    ``x`` is a list of C channels each of length L.
    """
    chans = [_vec(c, "channel") for c in x]
    C = len(chans)
    if C < 1:
        raise ValueError("need at least one channel")
    L = len(chans[0])
    if any(len(c) != L for c in chans):
        raise ValueError("all channels must be the same length")
    H = int(horizon)
    # time mixing, eq. (5): shared across channels, residual + norm
    mixed = []
    for c in chans:
        h = [_relu(t) for t in _addv(_matvec(wtime, c), btime)]
        mixed.append(_layernorm(_addv(c, h)))
    # feature mixing: the same MLP applied across the channel axis at
    # each time step, residual + norm (paper's prose)
    out = [[0.0] * L for _ in range(C)]
    for t in range(L):
        col = [mixed[i][t] for i in range(C)]
        h = [_relu(u) for u in _addv(_matvec(wfeat, col), bfeat)]
        newcol = _layernorm(_addv(col, h))
        for i in range(C):
            out[i][t] = newcol[i]
    # temporal projection, eq. (4)
    preds = [_addv(_matvec(wproj, out[i]), bproj) for i in range(C)]
    if any(len(p) != H for p in preds):
        raise ValueError("wproj must map L to the horizon")
    flat = [t for p in preds for t in p]
    return {"forecast": preds, "mixed": out, "nchannels": C, "L": L,
            "horizon": H, "mean": _mean(flat),
            "first": preds[0][0], "last": preds[-1][-1],
            "sumsq": sum(t * t for t in flat)}


def itrans(x, wembed, bembed, wq, wk, wv, wffn1, bffn1, wffn2, bffn2,
           wproj, bproj):
    """iTransformer: variates as tokens, attention across variates.

    Quoted from the paper:
        (1)  "h^0_n = Embedding(X_{:,n});
              H^{l+1} = TrmBlock(H^l), l = 0..L-1;
              Yhat_{:,n} = Projection(h^L_n)"
        (2)  "LayerNorm(H) = {[h_n - Mean(h_n)]/sqrt(Var(h_n))
                              | n = 1..N}"
        attention scores "A_{i,j} = (Q K^T / sqrt(d_k))_{i,j}"

    -- Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. and Long,
    M., "iTransformer: Inverted Transformers Are Effective for Time
    Series Forecasting", ICLR 2024 (arXiv:2310.06625).

    The inversion is the whole point: each VARIATE series becomes one
    token, so the attention matrix is N x N over variates rather than
    T x T over time steps.  All projections are caller-supplied.
    """
    chans = [_vec(c, "variate") for c in x]
    N = len(chans)
    if N < 1:
        raise ValueError("need at least one variate")
    T = len(chans[0])
    if any(len(c) != T for c in chans):
        raise ValueError("all variates must be the same length")
    # eq. (1) embedding: each whole series -> one token of width D
    toks = [_layernorm(_addv(_matvec(wembed, c), bembed)) for c in chans]
    D = len(toks[0])
    Q = [_matvec(wq, t) for t in toks]
    K = [_matvec(wk, t) for t in toks]
    V = [_matvec(wv, t) for t in toks]
    dk = len(Q[0])
    scores = [[sum(Q[i][d] * K[j][d] for d in range(dk)) / math.sqrt(dk)
               for j in range(N)] for i in range(N)]
    attn = [_softmaxv(row) for row in scores]
    ctx = [[sum(attn[i][j] * V[j][d] for j in range(N)) for d in range(len(V[0]))]
           for i in range(N)]
    # residual + LayerNorm (eq. 2), then the position-wise FFN
    h1 = [_layernorm(_addv(toks[i], ctx[i])) for i in range(N)]
    ffn = []
    for t in h1:
        u = [_relu(z) for z in _addv(_matvec(wffn1, t), bffn1)]
        ffn.append(_layernorm(_addv(t, _addv(_matvec(wffn2, u), bffn2))))
    preds = [_addv(_matvec(wproj, t), bproj) for t in ffn]
    flat = [t for p in preds for t in p]
    return {"forecast": preds, "attn": attn, "tokens": ffn, "nvariates": N,
            "T": T, "D": D, "horizon": len(preds[0]),
            "attndiag": sum(attn[i][i] for i in range(N)) / N,
            "mean": _mean(flat), "first": preds[0][0],
            "sumsq": sum(t * t for t in flat)}
