import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_uniform
from morie.fn._rrng_core import pgamma
from morie.fn._sci_core import gammaln
from morie.fn.demops import d8_flow_direction, flow_accumulation
from morie.fn.hydro import (
    baseflow_filter,
    breach_depressions,
    channel_slope,
    convolve_runoff,
    darcy_flow,
    flood_frequency,
    flow_duration_curve,
    height_above_drainage,
    hydrograph_summary,
    idf_fit,
    low_flow_frequency,
    meander_metrics,
    sample_lmoments,
    stream_segments,
    unit_hydrograph,
)

AMAX = [120, 95, 310, 180, 150, 220, 90, 260, 140, 175, 205, 130, 400, 160, 110]


def test_sample_lmoments_hand_and_gini_mean_difference():
    assert sample_lmoments([1, 2, 3, 4, 10]) == [4.0, 2.0, 0.5, 0.5]
    n = len(AMAX)
    gmd = ssum(abs(a - b) for i, a in enumerate(AMAX) for b in AMAX[i + 1 :]) / (n * (n - 1) / 2)
    assert abs(sample_lmoments(AMAX)[1] - gmd / 2) < 1e-12


def test_flood_frequency_quantiles_invert_the_fitted_cdf():
    l1, l2, t3, _ = sample_lmoments(AMAX)
    r = flood_frequency(AMAX, dist="gev")
    xi, al, k = r.params
    g = math.exp(gammaln(1 + k))
    assert abs(xi + al * (1 - g) / k - l1) < 1e-10 and abs(al * (1 - 2**-k) * g / k - l2) < 1e-10
    assert abs(2 * (1 - 3**-k) / (1 - 2**-k) - 3 - t3) < 1e-5
    for T, q in zip(r.return_periods, r.quantiles):
        assert abs(math.exp(-((1 - k * (q - xi) / al) ** (1 / k))) - (1 - 1 / T)) < 1e-12
    r = flood_frequency(AMAX, dist="gumbel")
    assert (
        abs(r.params[1] * math.log(2) - l2) < 1e-12
        and abs(r.params[0] + 0.57721566490153286 * r.params[1] - l1) < 1e-10
    )
    r = flood_frequency(AMAX, dist="lp3")
    m, s, g = r.params
    al, be = 4 / g**2, abs(0.5 * s * g)
    for T, q in zip(r.return_periods, r.quantiles):
        y = math.log10(q)
        F = float(pgamma((y - (m - al * be)) / be, al)) if g > 0 else 1 - float(pgamma(((m + al * be) - y) / be, al))
        assert abs(F - (1 - 1 / T)) < 1e-9
    assert round(flood_frequency(AMAX[:10], dist="gumbel").quantiles[0], 6) == 161.232848


def test_low_flow_minima_and_weibull_quantile():
    q = [5 + (i % 30) + 3 * (i // 365) + 2 * math.sin(i / 50) for i in range(365 * 8)]
    yrs = [i // 365 for i in range(365 * 8)]
    r = low_flow_frequency(q, yrs, d=7, T=10)
    for y, mval in enumerate(r.annual_minima):
        s = q[365 * y : 365 * (y + 1)]
        assert abs(mval - min(ssum(s[i : i + 7]) / 7 for i in range(359))) < 1e-12
    z, b, d = r.params
    assert abs(1 - math.exp(-(((r.quantile - z) / b) ** d)) - 0.1) < 1e-12
    r2 = low_flow_frequency(q, yrs, d=7, T=10, dist="lp3")
    assert min(r2.annual_minima) < r2.quantile < max(r2.annual_minima)


def test_flow_duration_curve_and_idf():
    r = flow_duration_curve([3, 1, 2, 4], [0.5, 0.1, 0.9])
    assert r.quantiles == [2.5, 4.0, 1.0] and r.exceedance == [0.2, 0.4, 0.6, 0.8]
    t = [5, 10, 15, 30, 60, 120]
    r = idf_fit(t, [1000 / (v + 8) ** 0.7 for v in t])
    assert abs(r.a - 1000) < 1e-6 and abs(r.b - 8) < 1e-8 and abs(r.c - 0.7) < 1e-10
    obs = [150, 118, 101, 72, 49, 31]
    r = idf_fit(t, obs)
    res = [o - f for o, f in zip(obs, r.fitted)]
    grad = [
        ssum(res[i] * g for i, g in enumerate(col))
        for col in (
            [f / r.a for f in r.fitted],
            [-r.c * f / (v + r.b) for f, v in zip(r.fitted, t)],
            [-f * math.log(v + r.b) for f, v in zip(r.fitted, t)],
        )
    ]
    assert max(abs(v) for v in grad) < 1e-6 * r.a


def test_unit_hydrographs_and_convolution():
    r = unit_hydrograph(method="scs", area=10, dt=1, D=2, tc=5 / 0.6)
    assert r.time_to_peak == 6.0 and abs(r.peak - 20.8 / 6) < 1e-15 and abs(r.base - 2.67 * 6) < 1e-12
    r = unit_hydrograph(method="nash", area=25, dt=0.5, D=1, n=3, k=2)
    vol = ssum(r.ordinates) * 0.5
    assert abs(vol - 25 / 0.36) < 1e-6 * 25 / 0.36
    assert convolve_runoff([1, 2], [0, 1, 3, 1, 0]) == [0.0, 1.0, 5.0, 7.0, 2.0, 0.0]


def test_baseflow_and_hydrograph_summary():
    q = [5, 5, 12, 30, 22, 15, 10, 8, 7, 6.5, 6.2, 6.0]
    r = baseflow_filter(q, passes=1)
    b, f = [q[0]], 0.0
    for t in range(1, len(q)):
        f = min(max(0.925 * f + 0.9625 * (q[t] - q[t - 1]), 0.0), q[t])
        b.append(q[t] - f)
    assert r.baseflow == b
    r3 = baseflow_filter(q)
    assert all(0 <= v <= w + 1e-12 for v, w in zip(r3.baseflow, q)) and 0 < r3.bfi < 1
    assert round(baseflow_filter([5, 5, 5, 5]).bfi, 12) == 1.0
    s = hydrograph_summary([1, 4, 8, 4, 2, 1, 0.5])
    assert (s.peak, s.time_to_peak) == (8.0, 2.0) and abs(s.recession_constant - 0.5) < 1e-14
    assert s.volume == 20.5 - 0.75


def _dem(n=9, seed=4):
    u = random_uniform(n * n, seed=seed, stream=0)
    return [[10 + 0.8 * i + 0.3 * abs(j - 4) + 2 * float(u[i * n + j]) for j in range(n)] for i in range(n)]


def test_stream_segments_and_horton():
    fd = [[2, 4, 8], [1, 4, 16], [1, 4, 16]]
    r = stream_segments(fd, flow_accumulation(fd), 1)
    assert r.counts == [7, 1] and abs(r.mean_lengths[0] - (5 + 2 * math.sqrt(2)) / 7) < 1e-15
    assert abs(r.bifurcation_ratio - 7) < 1e-12
    Z = breach_depressions(_dem(), epsilon=1e-3)
    fdr = d8_flow_direction(Z)
    s = stream_segments(fdr, flow_accumulation(fdr), 4)
    assert sum(s.counts) == len(s.segments) and all(seg["length"] >= 0 for seg in s.segments)


def test_channel_slope_and_meander():
    r = channel_slope([0, 100, 200], [10, 11, 14])
    assert (r.simple, r.equal_area) == (0.02, 0.015)
    assert abs(r.s1085 - (13.1 - 10.2) / 150) < 1e-15
    assert abs(r.harmonic - (200 / (100 / math.sqrt(0.01) + 100 / math.sqrt(0.03))) ** 2) < 1e-15
    th = [i * 0.2 for i in range(8)]
    m = meander_metrics([5 * math.cos(a) for a in th], [5 * math.sin(a) for a in th])
    assert all(abs(c - 0.2) < 1e-12 for c in m.curvature)
    xs = [i * 0.25 for i in range(41)]
    s = meander_metrics(xs, [math.sin(v) for v in xs])
    assert (
        abs(s.wavelength - 2 * math.pi) < 0.3
        and abs(s.sinuosity - s.length / math.dist((0, 0), (10, math.sin(10)))) < 1e-15
    )


def test_darcy_flow_linear_head():
    H = [[100 - 0.1 * 10 * j + 0.05 * 10 * (2 - i) for j in range(4)] for i in range(3)]
    r = darcy_flow(H, 3.0, res=10.0, porosity=0.25)
    assert all(abs(v - 0.3) < 1e-13 for row in r.qx for v in row)
    assert all(abs(v + 0.15) < 1e-13 for row in r.qy for v in row)
    assert abs(r.velocity[1][1] - math.hypot(0.3, 0.15) / 0.25) < 1e-13


def test_breach_and_hand_properties():
    G = _dem()
    Z = breach_depressions(G, epsilon=1e-3)
    n = len(G)
    assert all(Z[i][j] <= G[i][j] for i in range(n) for j in range(n))
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            assert min(Z[i + a][j + b] for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b) < Z[i][j]
    assert breach_depressions([[5, 5, 5, 5], [5, 1, 3, 5], [5, 5, 2, 5], [5, 5, 0, 5]])[2] == [5.0, 5.0, 1.0, 5.0]
    fd = d8_flow_direction(Z)
    ch = [[1 if v >= 6 else 0 for v in row] for row in flow_accumulation(fd)]
    h = height_above_drainage(Z, fd, ch, max_height=1.0, max_distance=2.0)
    for i in range(n):
        for j in range(n):
            if ch[i][j]:
                assert h.hand[i][j] == 0 and h.distance[i][j] == 0
            if h.hand[i][j] == h.hand[i][j]:
                assert h.hand[i][j] >= 0
                assert h.riparian[i][j] <= h.floodplain[i][j]
    r = height_above_drainage([[3, 2, 1]], [[1, 1, 0]], [[0, 0, 1]])
    assert (r.hand, r.distance) == ([[2.0, 1.0, 0.0]], [[2.0, 1.0, 0.0]])
