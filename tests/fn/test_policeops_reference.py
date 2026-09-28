"""policeops: hypercube against Erlang B/C and hand-solved chains; square-root law constants; Short model invariants."""

import math

import pytest

from morie.fn.policeops import hypercube_queue, short_crime_lattice, short_crime_pde, square_root_law


def erlang_b(N, a):
    b = 1.0
    for k in range(1, N + 1):
        b = a * b / (k + a * b)
    return b


def test_hypercube_matches_erlang_for_any_preferences():
    lam = [0.6, 0.9, 0.5]
    pref = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    r = hypercube_queue(lam, pref, service_rate=1.1)
    a = sum(lam) / 1.1
    assert r.loss == pytest.approx(erlang_b(3, a), abs=1e-12)
    assert sum(r.state_prob) == pytest.approx(1.0) and min(r.state_prob) >= 0
    # total workload equals carried load a (1 - loss) for the zero-line model
    assert sum(r.workload) == pytest.approx(a * (1 - r.loss), abs=1e-12)
    assert sum(map(sum, r.dispatch)) == pytest.approx(1 - r.loss, abs=1e-12)
    inf = hypercube_queue(lam, pref, service_rate=1.1, queue="infinite")
    B = erlang_b(3, a)
    C = 3 * B / (3 - a * (1 - B))
    assert inf.wait == pytest.approx(C, abs=1e-12)
    assert inf.mean_queue == pytest.approx(C * (a / 3) / (1 - a / 3), abs=1e-12)
    tt = hypercube_queue([1.0, 1.0], [[0, 1], [1, 0]], travel_time=[[2.0, 5.0], [6.0, 3.0]])
    d = tt.dispatch
    assert tt.mean_travel_time == pytest.approx(
        (2 * d[0][0] + 5 * d[0][1] + 6 * d[1][0] + 3 * d[1][1]) / sum(map(sum, d))
    )
    with pytest.raises(ValueError):
        hypercube_queue([5.0], [[0]], queue="infinite")


def test_square_root_law():
    assert square_root_law(64.0, 4.0).distance == pytest.approx(2.0)
    r = square_root_law(64.0, 4.0, metric="rectilinear", speed=0.5)
    assert r.distance == pytest.approx(math.sqrt(2 * math.pi) / 4 * 4) and r.time == pytest.approx(2 * r.distance)


def test_short_models():
    L = short_crime_lattice(6, 25, gamma=0.3, seed=4)
    assert len(L.burglaries) == 25 and all(v >= 0 for v in L.burglaries) and len(L.B) == 36
    Z = short_crime_lattice(5, 10, gamma=0.0)
    assert Z.burglaries == [0.0] * 10 and Z.B == [0.0] * 25
    # decay without crime: uniform B decays geometrically by (1 - omega)
    D0 = short_crime_lattice(3, 4, gamma=0.0, B0=[1.0] * 9, omega=0.1)
    assert pytest.approx([0.9**4] * 9) == D0.B
    eq = short_crime_pde([[0.03] * 5] * 5, [[0.002 / (1 / 30 + 0.03)] * 5] * 5, steps=200)
    assert eq.B[2][2] == pytest.approx(0.03, abs=1e-14) and eq.rho[1][3] == pytest.approx(eq.rho_star, abs=1e-14)
    # the transport term conserves offender mass: with no reaction terms the total is unchanged
    B = [[0.03 + 0.01 * math.sin(i + 2 * j) for j in range(6)] for i in range(6)]
    R = [[0.02 + 0.005 * math.cos(3 * i - j) for j in range(6)] for i in range(6)]
    nr = short_crime_pde(B, R, steps=5, omega=0.0, gamma=0.0, suppress=[[1] * 6] * 6)
    assert sum(map(sum, nr.rho)) == pytest.approx(sum(map(sum, R)), abs=1e-13)
