"""Dissimilarity-based compound selection: MaxMin and MaxSum."""
import importlib

import pytest

N = importlib.import_module("morie.fn.tncomp")
S = importlib.import_module("morie.fn.sasimi")

A = set(range(0, 20))
B = set(range(100, 120))
C = set(range(200, 220))
M = set(range(0, 18)) | set(range(100, 118)) | set(range(200, 218))
X = C - {219}
POOL = [A, B, C, M, X]
BIG = POOL + [set(range(300, 320)), set(range(400, 420)),
              set(range(500, 520)), set(range(0, 19)),
              set(range(100, 118))]


def test_the_fixture_separates_the_objectives():
    assert 1.0 - S.tanimoto(C, X) == pytest.approx(0.05)
    d = 1.0 - S.tanimoto(A, M)
    assert d == pytest.approx(1.0 - 18.0 / 56.0)
    assert 2.0 + 0.05 > 3.0 * d


def test_maxmin_takes_the_compromise_compound():
    assert sorted(N.maxmin_selection(POOL, 4, seed=0)) == [0, 1, 2, 3]


def test_maxsum_takes_the_near_duplicate():
    assert sorted(N.maxsum_selection(POOL, 4, seed=0)) == [0, 1, 2, 4]


def test_maxmin_has_the_larger_worst_distance():
    mm = N.diversity(POOL, N.maxmin_selection(POOL, 4, seed=0))
    ms = N.diversity(POOL, N.maxsum_selection(POOL, 4, seed=0))
    assert mm["min_distance"] > ms["min_distance"]


def test_the_worst_distance_never_improves_as_k_grows():
    prev = None
    for k in range(2, 9):
        d = N.diversity(BIG, N.maxmin_selection(BIG, k, seed=0))
        if prev is not None:
            assert d["min_distance"] <= prev + 1e-12
        prev = d["min_distance"]


def test_selecting_everything_returns_everything():
    assert sorted(N.maxmin_selection(BIG, len(BIG))) \
        == list(range(len(BIG)))


@pytest.mark.parametrize("k", list(range(1, 9)))
def test_the_selection_never_repeats(k):
    assert len(set(N.maxmin_selection(BIG, k, seed=2))) == k


def test_the_seed_is_honoured_and_reported():
    assert N.maxmin_selection(BIG, 3, seed=7)[0] == 7
    assert N.maxmin_diversity(BIG, 3, seed=7)["seed"] == 7


def test_selection_is_deterministic():
    assert N.maxmin_selection(BIG, 5, seed=0) \
        == N.maxmin_selection(BIG, 5, seed=0)


def test_the_distance_matrix_is_a_distance_matrix():
    D = N.distance_matrix(BIG)
    for i in range(len(D)):
        assert D[i][i] == 0.0
        for j in range(len(D)):
            assert D[i][j] == pytest.approx(D[j][i])


def test_the_default_seed_is_the_most_remote_compound():
    D = N.distance_matrix(BIG)
    tot = [sum(r) for r in D]
    assert N.maxmin_selection(BIG, 2)[0] == tot.index(max(tot))


@pytest.mark.parametrize("call", [
    lambda: N.distance_matrix([A]),
    lambda: N.maxmin_selection(POOL, 0),
    lambda: N.maxmin_selection(POOL, len(POOL) + 1),
    lambda: N.maxmin_selection(POOL, 3, seed=999),
    lambda: N.diversity(POOL, [1]),
    lambda: N.diversity(POOL, [1, 1]),
    lambda: N.maxmin_diversity(POOL, 3, "maxavg"),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
