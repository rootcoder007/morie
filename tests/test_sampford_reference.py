"""Sampford pi-ps sampling (sampling::UPsampford, UPsampfordpi2)."""

from morie.fn.sampfd import sampford_design

PIK = [0.1, 0.25, 0.35, 0.5, 0.8, 0.3, 0.7]


def test_joint_inclusion_matches_upsampfordpi2():
    J = sampford_design([0.2, 0.4, 0.6, 0.8]).extra["joint"]
    # sampling::UPsampfordpi2(c(0.2, 0.4, 0.6, 0.8))[1, ]
    ref = [0.2, 0.0277227722772277, 0.0534653465346534, 0.1188118811881188]
    assert all(abs(a - b) < 1e-14 for a, b in zip(J[0], ref))
    J = sampford_design(PIK).extra["joint"]
    # sum_{j != i} pi_ij = (n - 1) pi_i for a fixed-size design
    for i, p in enumerate(PIK):
        assert abs(sum(J[i][j] for j in range(7) if j != i) - 2 * p) < 1e-14
    assert abs(J[0][1] - 0.012824834888) < 1e-11 and abs(J[1][4] - 0.173454501561) < 1e-11


def test_samples_have_the_design_properties():
    counts = [0] * 7
    for s in range(200):
        r = sampford_design(PIK, seed=s)
        assert sum(r.value) == 3 and r.extra["attempts"] >= 1
        counts = [a + b for a, b in zip(counts, r.value)]
    # SampfordDesign(PIK, seed = s) for s in 0:199 in R: same Philox draws
    assert counts == [28, 48, 80, 110, 160, 48, 126]


def test_certainty_units():
    r = sampford_design([0.0, 1.0, 0.5, 0.5, 0.6, 0.4])
    J = r.extra["joint"]
    assert r.value[:2] == [0, 1] and sum(r.value) == 3
    assert J[1] == [0.0, 1.0, 0.5, 0.5, 0.6, 0.4] and J[0] == [0.0] * 6
    assert all(
        abs(sum(J[i][j] for j in range(6) if j != i) - 2 * p) < 1e-14
        for i, p in enumerate([0.0, 1.0, 0.5, 0.5, 0.6, 0.4])
    )
