"""Tests for causftbl (Pearl's front-door adjustment).

Replaces the generated stub, which imported
``causal_frontdoor_adjustment`` -- which does exist here, but the stub
called it with random numbers and asserted only a dict.
"""

from morie.fn.causftbl import causal_frontdoor_adjustment


def test_the_front_door_formula_on_a_deterministic_mediator():
    # X in {0,1}, Z copies X, Y copies Z: do(X=1) must give Y=1
    P_Z_X = [[1.0, 0.0], [0.0, 1.0]]          # P(Z|X)
    P_Y_XZ = [[[1.0, 0.0], [0.0, 1.0]],       # P(Y|X=0, Z)
              [[1.0, 0.0], [0.0, 1.0]]]       # P(Y|X=1, Z)
    P_X = [0.5, 0.5]
    res = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    assert abs(res["p_y_do_x"][1][1] - 1.0) < 1e-12
    assert abs(res["p_y_do_x"][0][1]) < 1e-12
    assert abs(res["ate"] - 1.0) < 1e-12


def test_a_mediator_that_ignores_x_gives_no_effect():
    P_Z_X = [[0.5, 0.5], [0.5, 0.5]]          # Z independent of X
    P_Y_XZ = [[[0.9, 0.1], [0.2, 0.8]],
              [[0.9, 0.1], [0.2, 0.8]]]
    P_X = [0.4, 0.6]
    res = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    assert abs(res["ate"]) < 1e-12


def test_each_row_of_the_result_is_a_distribution():
    P_Z_X = [[0.7, 0.3], [0.2, 0.8]]
    P_Y_XZ = [[[0.6, 0.4], [0.1, 0.9]],
              [[0.5, 0.5], [0.3, 0.7]]]
    P_X = [0.5, 0.5]
    res = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    for row in res["p_y_do_x"]:
        assert abs(sum(row) - 1.0) < 1e-12
        assert all(v >= -1e-15 for v in row)


def test_the_ate_is_the_difference_in_expected_outcome():
    P_Z_X = [[0.7, 0.3], [0.2, 0.8]]
    P_Y_XZ = [[[0.6, 0.4], [0.1, 0.9]],
              [[0.5, 0.5], [0.3, 0.7]]]
    P_X = [0.5, 0.5]
    res = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    assert abs(res["ate"] - (res["expected"][1] - res["expected"][0])) \
        < 1e-12


def test_validation():
    ok_z = [[0.5, 0.5], [0.5, 0.5]]
    ok_y = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]
    for call in (lambda: causal_frontdoor_adjustment([0.5, 0.5], ok_y,
                                                     [0.5, 0.5]),
                 lambda: causal_frontdoor_adjustment(ok_z, ok_y,
                                                     [0.5, 0.6]),
                 lambda: causal_frontdoor_adjustment([[0.5, 0.6],
                                                      [0.5, 0.5]], ok_y,
                                                     [0.5, 0.5])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
