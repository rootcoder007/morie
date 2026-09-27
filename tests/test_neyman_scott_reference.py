"""Neyman-Scott cluster processes (spatstat.random cluster model K and pcf)."""

from morie.fn.nsproc import neyman_scott_process

R = [0.02, 0.05, 0.1, 0.2]
# spatstatClusterModelInfo(m)$K / $pcf with kappa = 10 and scale 0.05
REF = {
    "thomas": (
        [0.005177693146204, 0.029973903326834, 0.094627982418754, 0.223832142254718],
        [4.058287770232, 3.478999886193, 2.170996630486, 1.058300489301],
    ),
    "cauchy": (
        [0.003198569492344, 0.018411262533983, 0.060705248417243, 0.180942346593596],
        [2.500617900352, 2.138820069467, 1.562697697598, 1.142352508683],
    ),
    "matern": (
        [0.01455130476675, 0.06650431447734, 0.1314159265359, 0.22566370614359],
        [10.511864337358, 5.978394872537, 1.0, 1.0],
    ),
}


def test_k_and_pcf_match_spatstat():
    for k, (K, g) in REF.items():
        r = neyman_scott_process(10.0, 5.0, 0.05, kernel=k, r=R)
        assert r.value == 50.0
        assert all(abs(a - b) < 1e-12 for a, b in zip(r.extra["K"], K))
        assert all(abs(a - b) < 1e-11 for a, b in zip(r.extra["pcf"], g))


def test_simulation_matches_r_arm_and_intensity():
    for k, counts, first in (
        ("thomas", [113, 94, 72], [0.382046066544, 0.087999212419]),
        ("cauchy", [94, 107, 109], [0.773795385777, 0.519382165482]),
        ("matern", [102, 96, 94], [0.008769044846, 0.218310541912]),
    ):
        r = neyman_scott_process(20.0, 4.0, 0.04, kernel=k, window=(0, 1, 0, 1), simulate=3, seed=1)
        assert [len(p) for p in r.extra["simulated"]] == counts
        assert all(abs(a - b) < 1e-11 for a, b in zip(r.extra["simulated"][0][0], first))
    sims = neyman_scott_process(20.0, 4.0, 0.04, window=(0, 1, 0, 1), simulate=100, seed=2).extra["simulated"]
    # Var N = lambda |W| (1 + mu) for small clusters: sd of the mean about 0.9
    assert abs(sum(len(p) for p in sims) / 100 - 80) < 4
