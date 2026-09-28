"""Tests for climidx: climate indices and climate-change diagnostics."""

import math

from morie.fn.climidx import (
    bcsd_downscale,
    cc_scaling,
    cyclone_energy,
    degree_heating_weeks,
    empirical_quantile_map,
    hadley_edge,
    hurricane_track,
    oni_index,
    pdo_index,
    probability_ratio,
    qbo_index,
    sea_level_semi_empirical,
)

SST = [27 + 1.3 * math.sin(2 * math.pi * i / 43) + 0.4 * math.cos(i * 0.9) for i in range(120)]


def _clim(x, start=1):
    c = []
    for m in range(12):
        v = [x[i] for i in range(len(x)) if (start - 1 + i) % 12 == m]
        c.append(sum(v) / len(v))
    return c


def test_oni_is_running_mean_of_monthly_anomalies():
    r = oni_index(SST, start_month=4)
    c = _clim(SST, 4)
    a = [SST[i] - c[(3 + i) % 12] for i in range(120)]
    for i in range(1, 119):
        assert abs(r.oni[i] - (a[i - 1] + a[i] + a[i + 1]) / 3) <= 1e-12
    assert math.isnan(r.oni[0]) and math.isnan(r.oni[-1])
    for i in range(120):
        if r.phase[i] == 1:
            assert r.oni[i] >= 0.5
        if r.phase[i] == -1:
            assert r.oni[i] <= -0.5


def test_oni_episode_needs_five_seasons():
    # a four-season warm spell is not an episode, a five-season one is
    x = [0.0] * 12 + [0.0] * 24
    base = oni_index(x, base=(0, 12)).phase
    assert set(base) == {0}
    y = [0.0] * 12 + [1.0] * 4 + [0.0] * 20
    assert set(oni_index(y, base=(0, 12)).phase) == {0}
    z = [0.0] * 12 + [1.0] * 5 + [0.0] * 19
    assert oni_index(z, base=(0, 12)).phase.count(1) == 5


def test_pdo_is_leading_eigenvector_and_standardised():
    X = [[math.sin(i * 0.3 + j) + 0.2 * j * math.cos(i * 0.11) for j in range(4)] for i in range(48)]
    r = pdo_index(X)
    cols = []
    for j in range(4):
        col = [X[t][j] for t in range(48)]
        c = _clim(col)
        cols.append([col[t] - c[t % 12] for t in range(48)])
    C = [[sum(cols[a][t] * cols[b][t] for t in range(48)) / 47 for b in range(4)] for a in range(4)]
    Ce = [sum(C[a][b] * r.loadings[b] for b in range(4)) for a in range(4)]
    for a in range(4):
        assert abs(Ce[a] - r.eigenvalue * r.loadings[a]) <= 1e-10
    assert abs(sum(v * v for v in r.loadings) - 1) <= 1e-12
    assert abs(sum(r.index)) <= 1e-9
    assert abs(sum(v * v for v in r.index) / 47 - 1) <= 1e-12
    assert r.loadings[max(range(4), key=lambda i: abs(r.loadings[i]))] < 0


def test_qbo_onsets_and_periods():
    u = [12 * math.sin(2 * math.pi * i / 27) + 3 * math.cos(i) for i in range(120)]
    r = qbo_index(u)
    c = _clim(u)
    a = [u[i] - c[i % 12] for i in range(120)]
    assert r.onsets == [i for i in range(1, 120) if a[i - 1] < 0 <= a[i]]
    assert r.periods == [r.onsets[i] - r.onsets[i - 1] for i in range(1, len(r.onsets))]


def test_dhw_accumulates_hotspots_of_at_least_one_degree():
    sst = [29 + 2 * math.sin(i / 9) for i in range(200)]
    r = degree_heating_weeks(sst, 29.3)
    hs = [max(v - 29.3, 0) for v in sst]
    for t in range(200):
        want = sum(h for h in hs[max(0, t - 83) : t + 1] if h >= 1) / 7
        assert abs(r.dhw[t] - want) <= 1e-12
    for h, d, a in zip(hs, r.dhw, r.alert):
        want = 0 if h <= 0 else (1 if h < 1 else (2 if d < 4 else (3 if d < 8 else 4)))
        assert a == want


def test_cyclone_energy_definitions():
    v = [20, 35, 50, 80, 110, 90, 60, 30]
    r = cyclone_energy(v)
    assert abs(r.ace - 1e-4 * sum(x * x for x in v if x >= 35)) <= 1e-12
    assert abs(r.pdi - sum(x**3 for x in v) * 21600) <= 1e-3


def test_hurricane_track_haversine():
    r = hurricane_track([0.0, 0.0], [0.0, 1.0])
    assert abs(r.distance[0] - 6371 * math.pi / 180) <= 1e-9
    assert abs(r.heading[0] - 90) <= 1e-12
    assert hurricane_track([12, 14, 17, 21], [-50, -55, -56, -52]).recurvature == 2


def test_hadley_edge_linear_crossing():
    lat = [-40.0, -20.0, 0.0, 20.0, 40.0]
    v = [[1.0, -2.0, 0.0, 2.0, -1.0], [0.0] * 5]
    r = hadley_edge(lat, [20000.0, 80000.0], v)
    c20, c40 = math.cos(math.radians(20)), math.cos(math.radians(40))
    want = 20 + 20 * 2 * c20 / (2 * c20 + c40)
    assert abs(r.edge_north - want) <= 1e-12
    assert abs(r.edge_south + want) <= 1e-12


def test_cc_scaling_recovers_exponential_rate():
    t = [float(i % 10) for i in range(60)]
    p = [math.exp(0.07 * v) for v in t]
    r = cc_scaling(t, p, min_count=2)
    assert abs(r.slope - 0.07) <= 1e-12
    tm = sum(t) / 60
    assert abs(r.cc_rate - 100 * 17.67 * 243.5 / (tm + 243.5) ** 2) <= 1e-12


def test_sea_level_recovers_rahmstorf_parameters():
    T = [0.1 * i for i in range(20)]
    H = [0.0]
    for i in range(19):
        H.append(H[-1] + 3.4 * ((T[i] + T[i + 1]) / 2 + 0.5))
    r = sea_level_semi_empirical(T, H, future_temp=[2.5])
    assert abs(r.a - 3.4) <= 1e-9 and abs(r.T0 + 0.5) <= 1e-9
    assert abs(r.projection[0] - (H[-1] + 3.4 * ((T[-1] + 2.5) / 2 + 0.5))) <= 1e-9


def test_quantile_map_maps_hist_quantiles_to_obs_quantiles():
    obs = [3.1, 5.2, 1.7, 8.8, 4.4, 6.0]
    hist = [1, 2, 2.5, 4, 7, 9, 10]
    so = sorted(obs)
    out = empirical_quantile_map(obs, hist, hist)
    for i, v in enumerate(out):
        h = (len(so) - 1) * i / 6
        lo = int(h)
        want = so[lo] + (h - lo) * (so[min(lo + 1, 5)] - so[lo])
        assert abs(v - want) <= 1e-12


def test_bcsd_idw_of_anomalies():
    r = bcsd_downscale([[1, 2, 3]], [[0, 1, 2]], [[1, 2]], [(0.0, 0.0)], [(3.0, 4.0)], [5.0])
    assert r.fine == [[5.0], [6.0]]
    r2 = bcsd_downscale(
        [[1, 2, 3], [2, 4, 6]], [[1, 2, 3], [2, 4, 6]], [[2, 3], [4, 6]], [(0, 0), (2, 0)], [(0.5, 0)], [10.0]
    )
    w = [0.5**-2, 1.5**-2]
    a = [[0, 0], [1, 2]]
    for t in range(2):
        assert abs(r2.fine[t][0] - (10 + (w[0] * a[t][0] + w[1] * a[t][1]) / sum(w))) <= 1e-12


def test_probability_ratio():
    r = probability_ratio([1, 5, 6, 7, 3], [1, 2, 3, 6], 4.5)
    assert abs(r.pr - (3 / 5) / (1 / 4)) <= 1e-12
    assert abs(r.far - (1 - (1 / 4) / (3 / 5))) <= 1e-12
