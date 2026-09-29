"""Tests for morie.fn.hotlg: Hotelling-Downs / Eaton-Lipsey equilibria."""

import pytest

from morie.fn._rng import random_uniform
from morie.fn.hotlg import cheatsheet, hotelling_model, hotellingmodel, hotlg
from morie.fn.plucmp import plurality_competition


def _markets(p):
    """Continuum-uniform market of each located point (pairs split), and each firm's longer half-market."""
    locs = sorted(set(p))
    out = {}
    for i, x in enumerate(locs):
        lo = 0.0 if i == 0 else (locs[i - 1] + x) / 2
        hi = 1.0 if i == len(locs) - 1 else (x + locs[i + 1]) / 2
        k = p.count(x)
        out[x] = ((hi - lo) / k, max(x - lo, hi - x) if k == 1 else (hi - lo) / 2)
    return out


def test_two_candidates_at_sample_median():
    r = hotlg(n_voters=41, n_candidates=2, seed=7)
    s = sorted(float(v) for v in random_uniform(41, seed=7))
    assert r.voter_median == s[20] and r.equilibrium_positions == [s[20], s[20]]
    assert r.max_gain == 0.0 and r.shares == [0.5, 0.5]


def test_three_candidates_have_no_pure_equilibrium():
    r = hotelling_model(n_candidates=3)
    assert r.has_pure_equilibrium is False and r.equilibrium_positions is None
    # converging all three at the median is not an equilibrium either
    v = [float(t) for t in random_uniform(100, seed=42)]
    assert max(plurality_competition(v, [r.voter_median] * 3).extra["gain"]) > 0


@pytest.mark.parametrize("n", [4, 5, 6, 7, 9])
def test_eaton_lipsey_conditions(n):
    fr = hotelling_model(n_voters=5, n_candidates=n).fractions
    a = 1 / (2 * n - 4)
    assert fr[0] == fr[1] == pytest.approx(a) and fr[-1] == fr[-2] == pytest.approx(1 - a)
    m = _markets(fr)
    assert min(v[0] for v in m.values()) >= max(v[1] for v in m.values()) - 1e-12
    assert sum(v[0] * fr.count(x) for x, v in m.items()) == pytest.approx(1.0, abs=1e-14)


def test_sample_positions_and_gain_recomputed():
    r = hotelling_model(n_voters=60, n_candidates=4, seed=3)
    s = sorted(float(v) for v in random_uniform(60, seed=3))
    q = []
    for f in (0.25, 0.25, 0.75, 0.75):
        i = 59 * f
        lo = int(i)
        q.append((1 - (i - lo)) * s[lo] + (i - lo) * s[lo + 1])
    assert max(abs(a - b) for a, b in zip(r.equilibrium_positions, q)) < 1e-15
    pc = plurality_competition(s, q)
    assert r.max_gain == pytest.approx(max(pc.extra["gain"]), abs=1e-15)
    assert hotellingmodel is hotelling_model and "Hotelling" in cheatsheet()
