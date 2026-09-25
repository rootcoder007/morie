"""Tests for spectral GRF simulation."""

from morie.fn import _array_core as np

from morie.fn.sgsps import sgsps


def test_sgsps_smoke():
    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 10)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    r = sgsps(coords, n_sims=2)
    assert r.name == "spectral_grf_sim"
    assert r.extra["simulations"].shape[0] == 2


def test_cheatsheet():
    from morie.fn.sgsps import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_sgsps_covariance_is_the_model():
    """Sample covariance over 3000 fields matches sill*exp(-h/range) +
    nugget*1(h=0) on every pair. Each entry is compared in units of its
    Monte Carlo standard error sqrt((C_aa C_bb + C_ab^2)/m), with a bound
    of 4.5 for 21 distinct entries; the old corner embedding with 1/N
    scaling gave variances near sill/N."""
    import math
    coords = [(0.5 * i, 0.5 * j) for j in range(2) for i in range(3)]
    sill, rng_, nug = 2.0, 1.2, 0.3
    r = sgsps(coords, "exponential", {"sill": sill, "range": rng_, "nugget": nug},
              n_sims=3000, seed=11)
    assert r.extra["embedding_exact"]
    sims = r.extra["simulations"].tolist()
    m, n = len(sims), len(coords)
    mean = [sum(s_[a] for s_ in sims) / m for a in range(n)]

    def c(a, b):
        return sill * math.exp(-math.dist(coords[a], coords[b]) / rng_) + (nug if a == b else 0.0)

    for a in range(n):
        for b in range(a, n):
            emp = sum((s_[a] - mean[a]) * (s_[b] - mean[b]) for s_ in sims) / (m - 1)
            se = math.sqrt((c(a, a) * c(b, b) + c(a, b) ** 2) / m)
            assert abs(emp - c(a, b)) < 4.5 * se


def test_sgsps_follows_coordinate_order():
    """Values are returned in the order of `coords`, not grid order."""
    coords = [(float(i), float(j)) for j in range(3) for i in range(2)]
    perm = [4, 0, 5, 2, 1, 3]
    a = sgsps(coords, n_sims=1, seed=3).extra["simulations"].tolist()[0]
    b = sgsps([coords[k] for k in perm], n_sims=1, seed=3).extra["simulations"].tolist()[0]
    assert b == [a[k] for k in perm]
