import math

from morie.fn._qpcore import ssum
from morie.fn._rng import random_uniform
from morie.fn.spboot import block_bootstrap_grid, bootstrap_bands, stationary_bootstrap, toroidal_shift_test

U = [float(v) for v in random_uniform(200, seed=41, stream=0)]
G = [[U[6 * i + j] + 0.3 * i for j in range(6)] for i in range(6)]


def test_block_bootstrap_values_come_from_blocks():
    r = block_bootstrap_grid(G, 2, nboot=40, seed=3)
    flat = sorted(v for row in G for v in row)
    assert abs(r.estimate - ssum(flat) / 36) < 1e-15 and r.se > 0
    one = block_bootstrap_grid(G, 6, nboot=5)
    assert all(abs(v - r.estimate) < 1e-12 for v in one.replicates)


def test_stationary_bootstrap_indices():
    x = [float(i) for i in range(20)]
    r = stationary_bootstrap(x, 0.2, nboot=30, seed=2)
    for ix in r.indices:
        breaks = sum(1 for a, b in zip(ix, ix[1:]) if b != (a + 1) % 20)
        assert len(ix) == 20 and breaks <= 19
    assert abs(r.estimate - 9.5) < 1e-15


def test_bands_and_toroidal_shift():
    reps = [[U[100 + 3 * r + k] for k in range(3)] for r in range(30)]
    b = bootstrap_bands(reps)
    assert all(lo <= hi for lo, hi in zip(b.pointwise_lower, b.pointwise_upper))
    assert all(sl <= e <= su for sl, e, su in zip(b.simultaneous_lower, b.estimate, b.simultaneous_upper))
    t = toroidal_shift_test(G, G)
    assert abs(t.statistic - 1) < 1e-12 and t.n_shifts == 35
    fa = [v for row in G for v in row]
    sh = [[G[(i + 1) % 6][(j + 2) % 6] for j in range(6)] for i in range(6)]
    fb = [v for row in sh for v in row]
    ma, mb = ssum(fa) / 36, ssum(fb) / 36
    c = ssum((a - ma) * (b - mb) for a, b in zip(fa, fb)) / math.sqrt(
        ssum((a - ma) ** 2 for a in fa) * ssum((b - mb) ** 2 for b in fb)
    )
    assert abs(t.simulated[6 + 2 - 1] - c) < 1e-12
