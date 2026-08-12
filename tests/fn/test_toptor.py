"""Tests for toptor (Nilakantan et al. 1987, topological torsion)."""

from morie.fn.toptor import (topological_torsion, topological_torsions,
                             toptor, torsion_similarity, trend_vector)


def _chain(n, order=1):
    return ["C"] * n, [(i, i + 1, order) for i in range(n - 1)]


def test_counts_on_small_graphs():
    assert len(topological_torsions(*_chain(3))) == 0
    t = topological_torsions(*_chain(4))
    assert sum(t.values()) == 1 and len(t) == 1
    assert sum(topological_torsions(*_chain(5)).values()) == 2
    assert sum(topological_torsions(*_chain(6)).values()) == 3
    star = (["C"] * 5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    assert sum(topological_torsions(*star).values()) == 0


def test_benzene_has_one_torsion_type():
    ring = (["C"] * 6, [(i, (i + 1) % 6, 1.5) for i in range(6)])
    bz = topological_torsions(*ring)
    assert len(bz) == 1
    assert sum(bz.values()) == 6
    code = list(bz)[0]
    assert all(a[0] == 1 and a[1] == "C" for a in code)
    assert [a[2] for a in code] == [1, 0, 0, 1]
    kekule = (["C"] * 6, [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1),
                          (4, 5, 2), (5, 0, 1)])
    assert set(topological_torsions(*kekule)) == set(bz)


def test_branch_counts_exclude_the_torsion():
    mb = (["C"] * 5, [(0, 1), (1, 2), (2, 3), (1, 4)])
    t = topological_torsions(*mb)
    want = ((0, "C", 0), (0, "C", 1), (0, "C", 0), (0, "C", 0))
    assert want in t or tuple(reversed(want)) in t
    assert sum(t.values()) == 2 and len(t) == 1


def test_uncommon_elements_become_y():
    weird = (["C", "N", "O", "Xx"], [(0, 1), (1, 2), (2, 3)])
    code = list(topological_torsions(*weird))[0]
    assert "Y" in [a[1] for a in code]


def test_similarity():
    a = topological_torsions(*_chain(6))
    assert torsion_similarity(a, a) == 1.0
    assert torsion_similarity({"p": 1, "q": 1},
                              {"q": 1, "r": 1, "s": 1}) == 0.4
    het = (["C", "N", "O", "S"], [(0, 1), (1, 2), (2, 3)])
    assert torsion_similarity(a, topological_torsions(*het)) == 0.0


def test_trend_vector_matches_the_formula():
    sets = [{"x": 1}, {"x": 1, "y": 1}, {"y": 1}, {"z": 1}]
    acts = [1.0, 2.0, 3.0, 4.0]
    tv = trend_vector(sets, acts, permutations=20, seed=1)
    mean = sum(acts) / 4
    for j, k in enumerate(tv["descriptors"]):
        want = sum((acts[i] - mean) * (1.0 if k in sets[i] else 0.0)
                   for i in range(4)) / 4.0
        assert abs(tv["vector"][j] - want) < 1e-12
    strong = trend_vector([{"a": 1}] * 5 + [{"b": 1}] * 5,
                          [0.0] * 5 + [10.0] * 5, 60, 2)
    assert strong["z"] > 3.0
    flat = trend_vector([{"a": 1, "b": 1}] * 10,
                        [0.0] * 5 + [10.0] * 5, 60, 3)
    assert flat["z"] == 0.0


def test_similarity_probe():
    probe = toptor([_chain(6)[0], _chain(4)[0]],
                   [_chain(6)[1], _chain(4)[1]], reference=_chain(5))
    assert abs(probe["similarity"][0] - 2.0 / 3.0) < 1e-12
    assert probe["ranking"][0] == 0


def test_validation():
    for call in (lambda: topological_torsions([], []),
                 lambda: topological_torsions(["C", "C"], [(0, 0)]),
                 lambda: topological_torsions(["C", "C"], [(0, 5)]),
                 lambda: topological_torsions(["C", "C"], [(0, 1, 0.5)]),
                 lambda: torsion_similarity({}, {})):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert topological_torsion is toptor
