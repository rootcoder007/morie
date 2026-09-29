"""Tests for smutl.simulate_utility_shocks: the Gumbel and normal transforms of the Philox uniforms."""

import math

from morie.fn._rng import random_uniform
from morie.fn.smutl import simulate_utility_shocks


def test_transforms():
    u = [float(v) for v in random_uniform(50, seed=4, stream=0)]
    g = simulate_utility_shocks(50, sigma=2.0, dist="gumbel", seed=4).value
    assert all(abs(a + 2.0 * math.log(-math.log(b))) < 1e-12 for a, b in zip(g, u))
    z = simulate_utility_shocks(50, dist="normal", seed=4).value
    # Phi(z) recovers the uniforms
    assert all(abs(0.5 * math.erfc(-a / math.sqrt(2)) - b) < 1e-12 for a, b in zip(z, u))
