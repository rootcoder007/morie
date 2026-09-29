import math

import pytest

from morie.fn.berkeley import berkeley_earth

ST = [(0.0, 0.0), (4.0, 0.5), (1.0, 3.0), (3.5, 3.5), (2.0, 1.5), (0.5, 2.0)]
TH = [math.sin(t / 3.0) for t in range(12)]
W = [[0.05 * math.sin(3.1 * i + 1.7 * t) for t in range(12)] for i in range(6)]
T = [[8.0 + i + TH[t] + W[i][t] for t in range(12)] for i in range(6)]


def test_theta_tracks_common_signal_and_is_centred():
    r = berkeley_earth(ST, series=T)
    m = sum(TH) / len(TH)
    for a, b in zip(r.theta, TH):
        assert abs(a - (b - m)) < 0.08
    assert abs(sum(r.theta)) < 1e-9
    for i in range(6):
        v = [T[i][t] - r.theta[t] for t in range(12)]
        assert r.baselines[i] == pytest.approx(sum(v) / len(v), abs=1e-9)


def test_weights_sum_to_one_constant_field():
    Tc = [[5.0 + i + TH[t] for t in range(12)] for i in range(6)]
    Tc[0][0] += 0.01
    r = berkeley_earth(ST, series=Tc)
    m = sum(TH) / len(TH)
    assert max(abs(a - (b - m)) for a, b in zip(r.theta, TH)) < 0.01
    assert 0 <= r.correlation_c <= 1 and r.correlation_length > 0


def test_missing_values_and_errors():
    Tm = [list(row) for row in T]
    Tm[2][3] = None
    Tm[4][7] = float("nan")
    r = berkeley_earth(ST, series=Tm, n_iter=3)
    assert len(r.theta) == 12 and all(math.isfinite(v) for v in r.theta)
    with pytest.raises(ValueError):
        berkeley_earth(ST)
