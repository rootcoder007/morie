"""Binary fingerprint similarity coefficients."""
import importlib
import itertools
import math

import pytest

S = importlib.import_module("morie.fn.sasimi")

A = {1, 3, 5, 7, 9, 11}
B = {1, 3, 5, 13, 15}
FPS = [set(c) for r in (1, 2, 3, 4)
       for c in itertools.combinations(range(6), r)]
PAIRS = list(itertools.combinations(FPS, 2))


def test_the_three_coefficients_on_a_hand_counted_pair():
    assert S.tanimoto(A, B) == pytest.approx(3.0 / 8.0)
    assert S.dice(A, B) == pytest.approx(6.0 / 11.0)
    assert S.cosine(A, B) == pytest.approx(3.0 / math.sqrt(30.0))


def test_tversky_specialises_to_tanimoto_and_dice():
    assert S.tversky(A, B, 1.0, 1.0) == pytest.approx(S.tanimoto(A, B))
    assert S.tversky(A, B, 0.5, 0.5) == pytest.approx(S.dice(A, B))


def test_tversky_measures_containment_asymmetrically():
    sub, sup = {1, 2}, set(range(1, 11))
    assert S.tversky(sub, sup, 1.0, 0.0) == 1.0
    assert S.tversky(sup, sub, 1.0, 0.0) == pytest.approx(0.2)
    assert S.tanimoto(sub, sup) == pytest.approx(0.2)


@pytest.mark.parametrize("f", [S.tanimoto, S.dice, S.cosine])
def test_endpoints(f):
    assert f(A, A) == pytest.approx(1.0)
    assert f({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_dice_and_cosine_never_fall_below_tanimoto():
    for x, y in PAIRS:
        t = S.tanimoto(x, y)
        assert S.dice(x, y) >= t - 1e-12
        assert S.cosine(x, y) >= t - 1e-12


@pytest.mark.parametrize("f", [S.tanimoto, S.dice, S.cosine])
def test_bounded_and_symmetric(f):
    for x, y in PAIRS:
        assert 0.0 <= f(x, y) <= 1.0 + 1e-12
        assert f(x, y) == pytest.approx(f(y, x))


def test_tanimoto_distance_is_a_metric():
    for x, y, z in itertools.combinations(FPS, 3):
        assert S.distance(x, z) <= S.distance(x, y) \
            + S.distance(y, z) + 1e-12


def test_dice_distance_is_not():
    bad = [t for t in itertools.combinations(FPS, 3)
           if 1 - S.dice(t[0], t[2])
           > (1 - S.dice(t[0], t[1])) + (1 - S.dice(t[1], t[2]))
           + 1e-12]
    assert bad


def test_tanimoto_carries_a_size_bias():
    assert S.tanimoto({1, 2}, {1, 2}) == 1.0
    assert S.tanimoto(set(range(1, 21)), {1, 2}) == pytest.approx(0.1)


def test_bit_vectors_and_index_sets_agree():
    assert S.fingerprint([0, 1, 0, 1, 1]) == frozenset({1, 3, 4})
    assert S.tanimoto([0, 1, 0, 1], [1, 1, 0, 0]) \
        == S.tanimoto({1, 3}, {0, 1})


def test_the_similarity_matrix():
    M = S.similarity_matrix([A, B, {1, 3, 5}])
    assert all(M[i][i] == 1.0 for i in range(3))
    assert all(M[i][j] == pytest.approx(M[j][i])
               for i in range(3) for j in range(3))


def test_nearest_neighbours_are_ordered():
    nn = S.nearest_neighbours({1, 3, 5}, [A, B, {1, 3, 5}, {20, 21}],
                              3)
    assert [x["index"] for x in nn] == [2, 1, 0]
    assert nn[0]["similarity"] == 1.0


def test_the_entry_point_reports_bit_counts():
    r = S.tanimoto_similarity(A, B)
    assert (r["bits_a"], r["bits_b"], r["bits_shared"]) == (6, 5, 3)
    assert r["similarity"] == pytest.approx(0.375)
    assert r["distance"] == pytest.approx(0.625)


@pytest.mark.parametrize("call", [
    lambda: S.tanimoto(set(), set()),
    lambda: S.fingerprint({-1}),
    lambda: S.fingerprint({9}, 4),
    lambda: S.tversky(A, B, -1.0, 1.0),
    lambda: S.similarity_matrix([A]),
    lambda: S.nearest_neighbours(A, [B], 0),
    lambda: S.distance(A, B, "euclidean"),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
