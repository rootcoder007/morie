r"""Largest Lyapunov exponent from a scalar time series.

Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993) "A practical
method for calculating largest Lyapunov exponents from small data sets",
*Physica D* 65(1-2), 117-134.

The attractor is reconstructed by delay embedding (the paper's eq. 5),

.. math::

   X_j = (x_j,\; x_{j+J},\; \ldots,\; x_{j+(m-1)J}),
   \qquad M = N - (m - 1)J,

each point's nearest neighbour is found subject to a temporal separation
larger than the mean period (eqs. 7 and 8),

.. math::

   d_j(0) = \min_{X_{j'}} \lVert X_j - X_{j'} \rVert,
   \qquad |j - j'| > \text{mean period},

and the neighbours are then assumed to separate at the rate of the largest
exponent, :math:`d_j(i) \approx C_j e^{\lambda_1 (i \Delta t)}` (eq. 11).
Taking logs (eq. 12) turns that into a family of roughly parallel lines,
and the exponent is the slope of their average (eq. 13),

.. math::

   y(i) = \frac{1}{\Delta t} \bigl\langle \ln d_j(i) \bigr\rangle_j .

The averaging over :math:`j` is what makes the estimate work on short
series -- "the method is accurate for small data sets because it takes
advantage of all the available data" -- and the paper's point about
:math:`C_j` is that no normalisation by :math:`d_j(0)` is needed, since a
constant offset does not change a slope.

Three routes are available, all printed in the paper, and all reachable
through ``method``:

``"rosenstein"`` (default)
    Equation 13. Least squares on :math:`y(i)` over the initial rise.

``"sato"``
    Sato et al.'s eq. 9, :math:`\lambda_1(i) = \frac{1}{i \Delta t}
    \frac{1}{M - i} \sum_j \ln \frac{d_j(i)}{d_j(0)}`, read at the end of
    the fitting window.

``"sato_k"``
    Sato et al.'s eq. 10, :math:`\lambda_1(i, k) = \frac{1}{k \Delta t}
    \frac{1}{M - k} \sum_j \ln \frac{d_j(i + k)}{d_j(i)}`, with
    :math:`\lambda_1` read off the plateau in :math:`i`. Rosenstein et al.
    report that "locating this plateau is sometimes problematic, and the
    resulting estimates of :math:`\lambda_1` are unreliable"; it is here
    because the paper prints it, not because it is recommended.

Table 1 of the paper gives the expected exponents this module is anchored
against: 0.693 for the logistic map at :math:`\mu = 4` and 0.418 for the
Henon map at :math:`a = 1.4`, :math:`b = 0.3`.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "lyapunov_exponent",
    "largest_lyapunov",
    "embed",
    "autocorrelation_lag",
    "mean_period",
    "divergence_curve",
]


def _as_series(y):
    out = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float))]
    if len(out) < 10:
        raise ValueError("lyapun: need at least 10 observations, got %d"
                         % len(out))
    for v in out:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("lyapun: the series contains a non-finite "
                             "value")
    return out


def embed(y, m, tau):
    r"""Delay embedding, the paper's eq. 5.

    Returns the :math:`M = N - (m - 1)J` reconstructed points, in order.
    """
    y = _as_series(y)
    m = int(m)
    tau = int(tau)
    if m < 1:
        raise ValueError("lyapun: the embedding dimension must be >= 1")
    if tau < 1:
        raise ValueError("lyapun: the reconstruction delay must be >= 1")
    n_pts = len(y) - (m - 1) * tau
    if n_pts < 3:
        raise ValueError("lyapun: m = %d and J = %d leave only %d "
                         "reconstructed points" % (m, tau, n_pts))
    return [[y[j + k * tau] for k in range(m)] for j in range(n_pts)]


def autocorrelation_lag(y, threshold=None):
    r"""The lag where the autocorrelation first falls to :math:`1 - 1/e`
    of its initial value.

    "We have found a good approximation of J to equal the lag where the
    autocorrelation function drops to 1 - 1/e of its initial value."
    """
    y = _as_series(y)
    n = len(y)
    if threshold is None:
        threshold = 1.0 - 1.0 / math.e
    mu = sum(y) / n
    c0 = sum((v - mu) ** 2 for v in y) / n
    if c0 <= 0.0:
        raise ValueError("lyapun: the series is constant, so no delay can "
                         "be chosen from its autocorrelation")
    for lag in range(1, n):
        c = sum((y[t] - mu) * (y[t + lag] - mu) for t in range(n - lag)) / n
        if c / c0 <= threshold:
            return lag
    return 1


def mean_period(y, dt=1.0):
    r"""Reciprocal of the mean frequency of the power spectrum.

    The paper's footnote on eq. 8: the mean period is estimated "as the
    reciprocal of the mean frequency of the power spectrum", the mean
    being taken with the power as the weight. Returned in samples, which
    is what the neighbour search needs.
    """
    y = _as_series(y)
    n = len(y)
    mu = sum(y) / n
    spec = np.fft.rfft([v - mu for v in y])
    freqs = np.fft.rfftfreq(n, dt)
    power = [float(abs(c)) ** 2 for c in spec]
    wsum = sum(power[1:])
    if wsum <= 0.0:
        return 1.0
    f_mean = sum(float(freqs[k]) * power[k]
                 for k in range(1, len(power))) / wsum
    if f_mean <= 0.0:
        return float(n)
    return (1.0 / f_mean) / dt


def _nearest_neighbours(pts, min_sep):
    r"""Eq. 7 with the constraint of eq. 8.

    Returns ``(index, d0)`` per point; ``index`` is -1 where the temporal
    constraint leaves no candidate at all.
    """
    n_pts = len(pts)
    m = len(pts[0])
    nn = [-1] * n_pts
    d0 = [0.0] * n_pts
    for j in range(n_pts):
        best, best_d = -1, float("inf")
        pj = pts[j]
        for jp in range(n_pts):
            if abs(j - jp) <= min_sep:
                continue
            pk = pts[jp]
            s = 0.0
            for k in range(m):
                diff = pj[k] - pk[k]
                s += diff * diff
                if s >= best_d:
                    break
            if s < best_d:
                best_d, best = s, jp
        nn[j] = best
        d0[j] = math.sqrt(best_d) if best >= 0 else float("nan")
    return nn, d0


def _distance(pts, a, b):
    s = 0.0
    for k in range(len(pts[a])):
        diff = pts[a][k] - pts[b][k]
        s += diff * diff
    return math.sqrt(s)


def divergence_curve(y, m=None, tau=None, dt=1.0, min_sep=None,
                     max_steps=None):
    r"""The paper's :math:`y(i) = \langle \ln d_j(i) \rangle`, and the
    pieces every route is built from.

    ``min_sep`` defaults to the mean period in samples (eq. 8); pass 0 to
    drop the constraint, which the paper warns against because then a
    point's nearest neighbour is simply its temporal neighbour and the
    pair are not "nearby initial conditions for different trajectories".
    """
    y = _as_series(y)
    n = len(y)
    if tau is None:
        tau = autocorrelation_lag(y)
    if m is None:
        m = 3
    if dt <= 0:
        raise ValueError("lyapun: the sampling period must be positive")
    pts = embed(y, m, tau)
    n_pts = len(pts)
    if min_sep is None:
        min_sep = int(round(mean_period(y, dt)))
    min_sep = int(min_sep)
    if min_sep < 0:
        raise ValueError("lyapun: min_sep must be >= 0")
    if min_sep >= n_pts - 2:
        raise ValueError("lyapun: the mean period (%d samples) leaves no "
                         "admissible neighbours among %d reconstructed "
                         "points; pass min_sep explicitly"
                         % (min_sep, n_pts))
    nn, d0 = _nearest_neighbours(pts, min_sep)
    usable = [j for j in range(n_pts) if nn[j] >= 0 and d0[j] > 0.0]
    if len(usable) < 3:
        raise ValueError("lyapun: fewer than three usable neighbour pairs")
    if max_steps is None:
        max_steps = max(1, n_pts // 4)
    max_steps = int(max_steps)

    times, curve, counts, ratio = [], [], [], []
    for i in range(0, max_steps + 1):
        tot, tot_ratio, cnt = 0.0, 0.0, 0
        for j in usable:
            jp = nn[j]
            if j + i >= n_pts or jp + i >= n_pts:
                continue
            d = _distance(pts, j + i, jp + i)
            if d <= 0.0:
                continue
            tot += math.log(d)
            tot_ratio += math.log(d / d0[j])
            cnt += 1
        if cnt == 0:
            break
        times.append(i * dt)
        curve.append(tot / cnt)
        ratio.append(tot_ratio / cnt)
        counts.append(cnt)
    return {
        "time": times,
        "log_divergence": curve,     # <ln d_j(i)>, eq. 13 before 1/dt
        "log_ratio": ratio,          # <ln (d_j(i)/d_j(0))>, for eq. 9
        "n_pairs": counts,
        "neighbour": nn,
        "d0": d0,
        "points": pts,
        "m": m,
        "tau": tau,
        "min_sep": min_sep,
        "n_points": n_pts,
        "n_obs": n,
    }


def _linear_region(curve, lo_frac=0.1, hi_frac=0.8):
    """The straight middle of the divergence curve.

    The paper fits "the initial linear rise" of eq. 13 by eye. Two things
    sit outside that rise and both bias the slope if they are fitted: a
    short transient at the start, where the neighbours have not yet
    aligned with the most unstable direction, and the plateau at the end,
    where they are as far apart as the attractor allows. The plateau is by
    far the more damaging of the two.

    The window returned here is the stretch over which the curve climbs
    from ``lo_frac`` to ``hi_frac`` of its total climb. It has no transient
    and no plateau in it by construction, and it needs no notion of a
    "steep enough" step, which is what makes it work equally on a map
    sampled once per iteration and on a flow sampled a hundred times per
    period.

    This is a heuristic standing in for reading the plot, and the answer
    does depend on it -- see the module docstring. Pass ``fit`` to choose
    the window yourself; ``log_divergence`` is returned so the choice can
    be seen.
    """
    n = len(curve)
    if n < 4:
        return 0, n
    c_lo, c_hi = min(curve), max(curve)
    span = c_hi - c_lo
    if span <= 0.0:
        return 0, n
    top = curve.index(c_hi)
    if top < 3:
        return 0, n
    lo_level = c_lo + lo_frac * span
    hi_level = c_lo + hi_frac * span
    lo = 0
    while lo < top and curve[lo] < lo_level:
        lo += 1
    hi = lo
    while hi < top and curve[hi] < hi_level:
        hi += 1
    hi = min(hi + 1, n)
    if hi - lo < 3:
        lo, hi = 0, n
    return lo, hi


def _ols_slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx <= 0:
        raise ValueError("lyapun: the fitting window has no spread in "
                         "time")
    sxy = sum((xs[k] - mx) * (ys[k] - my) for k in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [ys[k] - intercept - slope * xs[k] for k in range(n)]
    sse = sum(r * r for r in resid)
    sst = sum((v - my) ** 2 for v in ys)
    se = (math.sqrt(sse / (n - 2) / sxx) if n > 2 and sse > 0 else 0.0)
    return slope, intercept, se, (1.0 - sse / sst if sst > 0 else 1.0)


def lyapunov_exponent(y, embedding=None, tau=None, dt=1.0, fit=None,
                      min_sep=None, max_steps=None, method="rosenstein",
                      k=None):
    r"""Largest Lyapunov exponent, Rosenstein et al. (1993).

    Parameters
    ----------
    y : sequence of float
        The scalar time series.
    embedding : int, optional
        Embedding dimension :math:`m`. Defaults to 3. The paper's section
        4.1: results are satisfactory only once :math:`m` is at least the
        topological dimension of the system, because "chaotic systems are
        effectively stochastic when embedded in a phase space that is too
        small to accommodate the true dynamics".
    tau : int, optional
        Reconstruction delay :math:`J`. Defaults to the lag where the
        autocorrelation falls to :math:`1 - 1/e` of its initial value.
    dt : float
        Sampling period. The exponent is per unit of ``dt``.
    fit : (int, int), optional
        Half-open range of steps :math:`i` used for the least-squares fit,
        i.e. the initial linear rise before the curve saturates. Defaults
        to the stretch over which the curve climbs from 10% to 80% of its
        total climb -- a heuristic stand-in for the paper's reading of the
        plot by eye, with neither the initial transient nor the final
        plateau in it.
    min_sep : int, optional
        Minimum temporal separation of a neighbour pair (eq. 8). Defaults
        to the mean period in samples.
    method : {"rosenstein", "sato", "sato_k"}
        Which of the paper's three routes to report as ``estimate``. All
        three are computed and returned.
    k : int, optional
        The ``k`` of eq. 10. Defaults to the width of the fitting window.

    Returns
    -------
    RichResult
        ``estimate`` is the exponent from the chosen route.
    """
    if method not in ("rosenstein", "sato", "sato_k"):
        raise ValueError("lyapun: method must be 'rosenstein', 'sato' or "
                         "'sato_k'")
    dv = divergence_curve(y, m=embedding, tau=tau, dt=dt, min_sep=min_sep,
                          max_steps=max_steps)
    times = dv["time"]
    curve = dv["log_divergence"]
    n_steps = len(curve)
    if fit is None:
        lo, hi = _linear_region(curve)
    else:
        lo, hi = int(fit[0]), int(fit[1])
        if lo < 0 or hi > n_steps or hi - lo < 2:
            raise ValueError("lyapun: the fitting window must lie inside "
                             "0..%d and span at least two steps"
                             % n_steps)
    slope, intercept, se, r2 = _ols_slope(times[lo:hi], curve[lo:hi])

    # eq. 9, read at the end of the same window
    i_end = hi - 1
    sato = (dv["log_ratio"][i_end] / (i_end * dt)) if i_end > 0 else \
        float("nan")

    # eq. 10, on the same window
    if k is None:
        k = max(1, hi - lo)
    k = int(k)
    sato_k, sato_k_curve = float("nan"), []
    if k >= 1 and n_steps > k:
        pts = dv["points"]
        nn, n_pts = dv["neighbour"], dv["n_points"]
        usable = [j for j in range(n_pts) if nn[j] >= 0 and dv["d0"][j] > 0]
        for i in range(0, n_steps - k):
            tot, cnt = 0.0, 0
            for j in usable:
                jp = nn[j]
                if max(j, jp) + i + k >= n_pts:
                    continue
                d_i = _distance(pts, j + i, jp + i)
                d_ik = _distance(pts, j + i + k, jp + i + k)
                if d_i <= 0.0 or d_ik <= 0.0:
                    continue
                tot += math.log(d_ik / d_i)
                cnt += 1
            if cnt == 0:
                break
            sato_k_curve.append(tot / cnt / (k * dt))
        if sato_k_curve:
            # The "plateau", searched only inside the fitting window: past
            # it the curve has saturated and its flattest stretch is the
            # zero tail, which would be reported as "no chaos" for every
            # chaotic system.
            search = sato_k_curve[:max(3, min(hi, len(sato_k_curve)))]
            w = max(2, len(search) // 4)
            best, best_var = 0, float("inf")
            for s in range(0, len(search) - w + 1):
                seg = search[s:s + w]
                mu = sum(seg) / w
                var = sum((v - mu) ** 2 for v in seg) / w
                if var < best_var:
                    best_var, best = var, s
            seg = search[best:best + w]
            sato_k = sum(seg) / len(seg)

    estimate = {"rosenstein": slope, "sato": sato,
                "sato_k": sato_k}[method]
    return RichResult(payload={
        "estimate": estimate,
        "lambda1": estimate,
        "rosenstein": slope,
        "sato": sato,
        "sato_k": sato_k,
        "sato_k_curve": sato_k_curve,
        "se": se,
        "r_squared": r2,
        "intercept": intercept,
        "time": times,
        "log_divergence": curve,
        "log_ratio": dv["log_ratio"],
        "n_pairs": dv["n_pairs"],
        "fit_range": (lo, hi),
        "k": k,
        "m": dv["m"],
        "tau": dv["tau"],
        "min_sep": dv["min_sep"],
        "n_points": dv["n_points"],
        "n": dv["n_obs"],
        "dt": dt,
        "method": ("largest Lyapunov exponent, Rosenstein, Collins & "
                   "De Luca (1993), route '%s'" % method),
        "note": ("the exponent is the slope of <ln d_j(i)> over the "
                 "initial rise; a positive value indicates chaos, and the "
                 "fitting window is the caller's to choose because the "
                 "curve saturates once the neighbours are as far apart as "
                 "the attractor allows"),
    })


largest_lyapunov = lyapunov_exponent


def cheatsheet():
    return ("lyapun: largest Lyapunov exponent (Rosenstein, Collins & De "
            "Luca 1993). Embed with delay J and dimension m, find each "
            "point's nearest neighbour at least a mean period away, and "
            "take lambda_1 as the slope of <ln d_j(i)> against i*dt over "
            "the initial rise -- no normalisation by d_j(0) is needed, "
            "since a constant offset does not change a slope. Expected "
            "values from the paper's table 1: 0.693 for the logistic map "
            "at mu = 4, 0.418 for the Henon map. Routes: 'rosenstein' "
            "(eq. 13, default), 'sato' (eq. 9), 'sato_k' (eq. 10, whose "
            "plateau the paper itself calls unreliable).")
