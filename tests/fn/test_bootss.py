"""Tests for bootss (Rao-Wu rescaled survey bootstrap).

Replaces the generated stub, which imported ``bootstrap_survey``.
"""

from morie.fn.bootss import bootss


def _design(n_strata=4, n_clusters=5):
    y, w, strata, clusters = [], [], [], []
    for h in range(n_strata):
        for c in range(n_clusters):
            for k in range(3):
                y.append(10.0 + h + c * 0.5 + k * 0.1)
                w.append(2.0)
                strata.append(h)
                clusters.append("h%d_c%d" % (h, c))
    return y, w, strata, clusters


def test_point_estimate_is_the_weighted_total():
    y, w, s, c = _design()
    res = bootss(y, w, s, c, B=50, seed=1)
    want = sum(y[i] * w[i] for i in range(len(y)))
    assert abs(res["estimate"] - want) < 1e-9


def test_a_custom_statistic_is_used_when_given():
    y, w, s, c = _design()
    mean = bootss(y, w, s, c, B=20, seed=1,
                  statistic=lambda yy, ww: sum(a * b for a, b in
                                               zip(yy, ww)) / sum(ww))
    assert abs(mean["estimate"] -
               sum(y[i] * w[i] for i in range(len(y))) / sum(w)) < 1e-9


def test_variance_is_positive_and_matches_its_replicates():
    y, w, s, c = _design()
    res = bootss(y, w, s, c, B=100, seed=1)
    assert res["variance"] > 0
    assert abs(res["se"] - res["variance"] ** 0.5) < 1e-12
    assert len(res["replicates"]) == 100
    assert res["n_strata"] == 4


def test_more_between_cluster_spread_gives_a_larger_se():
    y, w, s, c = _design()
    spread = [v + 5.0 * int(c[i][-1]) for i, v in enumerate(y)]
    tight = bootss(y, w, s, c, B=100, seed=1)["se"]
    loose = bootss(spread, w, s, c, B=100, seed=1)["se"]
    assert loose > tight


def test_the_same_seed_reproduces_the_replicates():
    y, w, s, c = _design()
    a = bootss(y, w, s, c, B=40, seed=7)["replicates"]
    b = bootss(y, w, s, c, B=40, seed=7)["replicates"]
    assert a == b


def test_validation():
    y, w, s, c = _design()
    for call in (lambda: bootss(y[:-1], w, s, c),
                 lambda: bootss(y, [0.0] * len(y), s, c),
                 lambda: bootss([1.0, 2.0], [1.0, 1.0], [0, 0],
                                ["a", "a"])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
