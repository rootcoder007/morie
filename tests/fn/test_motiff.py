"""Tests for motiff (network-motif significance).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.motiff import motiff


def _ffl_network(n_extra=0):
    # X -> Y, Y -> Z, X -> Z is one feed-forward loop
    n = 3 + n_extra
    A = [[0] * n for _ in range(n)]
    A[0][1] = A[1][2] = A[0][2] = 1
    return A


def test_a_single_feed_forward_loop_is_counted_once():
    res = motiff(_ffl_network(), motif="ffl", n_random=20, seed=1)
    assert res["count"] == 1


def test_a_three_cycle_is_counted_as_a_cycle_not_an_ffl():
    A = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    assert motiff(A, motif="cycle3", n_random=20, seed=1)["count"] == 1
    assert motiff(A, motif="ffl", n_random=20, seed=1)["count"] == 0


def test_an_empty_network_has_no_motifs():
    A = [[0] * 5 for _ in range(5)]
    res = motiff(A, motif="ffl", n_random=20, seed=1)
    assert res["count"] == 0


def test_a_layered_network_keeps_its_ffls_under_the_null():
    # A three-layer feed-forward network has 27 FFLs -- and this is the
    # informative case: degree-preserving randomisation cannot destroy
    # them, because every rewiring of a layered DAG that keeps each
    # node's in- and out-degree is again layered. The null model has
    # nothing to say here, so z is 0 and p is 1. A motif count means
    # nothing without knowing what the null could have produced.
    n = 9
    A = [[0] * n for _ in range(n)]
    for i in range(3):
        for j in range(3, 6):
            A[i][j] = 1
        for k in range(6, 9):
            A[i][k] = 1
    for j in range(3, 6):
        for k in range(6, 9):
            A[j][k] = 1
    res = motiff(A, motif="ffl", n_random=60, seed=1)
    assert res["count"] == 27
    assert abs(res["rand_mean"] - 27.0) < 1e-9
    assert res["rand_sd"] == 0.0
    assert res["z_score"] == 0.0
    assert res["p_value"] == 1.0


def test_randomisation_summaries_are_reported():
    res = motiff(_ffl_network(3), motif="ffl", n_random=40, seed=1)
    assert res["n_random"] == 40
    assert res["rand_sd"] >= 0.0
    assert res["rand_mean"] >= 0.0


def test_seed_reproducibility():
    a = motiff(_ffl_network(3), n_random=30, seed=5)["z_score"]
    b = motiff(_ffl_network(3), n_random=30, seed=5)["z_score"]
    assert a == b


def test_validation():
    for call in (lambda: motiff([[0, 1], [1, 0]]),
                 lambda: motiff([[0, 1, 0], [1, 0, 0]]),
                 lambda: motiff(_ffl_network(), motif="clique4")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
