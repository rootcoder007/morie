"""method="em" reaches the direct method's optimum, and no longer at minutes per fit.

Round-8 finding: EM took 20-140 s where every other method took under 1 s on 1,083 events,
because each M-step ran a full inner optimisation (every evaluation an O(n^2) pair pass) and
the polish used the full O(n^2) Lomax sum. Generalised EM with a short inner loop, a hand-over
to the kernel's own direct route, and the same optimum.
"""

import math
import random
import time

import numpy as np
import pytest

from morie.tps_hawkes_advanced import fit_hawkes_general


def _simulate(n_target=400, seed=3):
    random.seed(seed)
    mu, alpha, beta = 0.3, 0.5, 1.0
    T = n_target / 0.6
    t, ev = 0.0, []
    while t < T:
        lam_bar = mu + sum(alpha * math.exp(-beta * (t - s)) for s in ev[-200:])
        t += random.expovariate(lam_bar)
        lam = mu + sum(alpha * math.exp(-beta * (t - s)) for s in ev[-200:])
        if random.random() < lam / lam_bar:
            ev.append(t)
    return np.array([e for e in ev if e < T]), T


@pytest.mark.parametrize(
    ("kernel", "baseline"), [("exponential", "constant"), ("gamma", "constant"), ("lomax", "sinusoidal")]
)
def test_em_matches_the_direct_optimum_quickly(kernel, baseline):
    pytest.importorskip("morie._core")
    ev, T = _simulate()
    direct = fit_hawkes_general(ev, T, kernel, baseline)
    t0 = time.perf_counter()
    em = fit_hawkes_general(ev, T, kernel, baseline, method="em")
    took = time.perf_counter() - t0
    assert em["nll"] == pytest.approx(direct["nll"], abs=1e-6)
    assert took < 60, f"EM took {took:.1f}s on {len(ev)} events"
