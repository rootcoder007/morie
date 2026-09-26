"""Aalen-Johansen CIF and variance against R cmprsk::cuminc.

Reference: cuminc(t, e, cencode = 0)[["1 1"]] (est, var) on the
deterministic data below (ties on purpose, censoring between events).
"""

import math

from morie import survival

REF = {1.2: (0.0168067226890756, 0.000140036074250238), 5.1: (0.201430458174108, 0.00142851408846308),
       8.0: (0.389754364614983, 0.00221270696282109), 9.9: (0.704812572419722, 0.00237076338475839)}


def test_cif_matches_cuminc():
    t, e = [], []
    for i in range(120):
        t.append(round(1 + 9 * abs(math.sin(1.37 * i)), 1))
        e.append([0, 1, 2, 1][int(4 * abs(math.cos(2.1 * i))) % 4])
    r = survival.cumulative_incidence_function(t, e, event_of_interest=1)
    got = {round(float(tt), 1): (float(f), float(v)) for tt, f, v in zip(r.times, r.cif, r.variance)}
    for k, (f, v) in REF.items():
        assert abs(got[k][0] - f) <= 1e-13
        assert abs(got[k][1] - v) <= 1e-15
