"""Tests for linkmm (IRT mean/mean and mean/sigma linking).

Replaces the generated stub, which imported ``linking_meanmean``.
"""

from morie.fn.linkmm import linkmm


def test_mean_mean_recovers_a_known_transformation():
    # apply b* = A b + B, a* = a / A with A = 2, B = 0.5 and check the
    # method reads A and B back off the two item sets
    a_from = [1.0, 0.8, 1.2, 0.9]
    b_from = [-1.0, 0.0, 1.0, 0.5]
    A, B = 2.0, 0.5
    a_to = [a / A for a in a_from]
    b_to = [A * b + B for b in b_from]
    res = linkmm(a_from, b_from, a_to, b_to)
    assert abs(res["A"] - A) < 1e-9
    assert abs(res["B"] - B) < 1e-9


def test_mean_sigma_recovers_it_too():
    a_from = [1.0, 0.8, 1.2, 0.9]
    b_from = [-1.0, 0.0, 1.0, 0.5]
    A, B = 1.5, -0.25
    a_to = [a / A for a in a_from]
    b_to = [A * b + B for b in b_from]
    res = linkmm(a_from, b_from, a_to, b_to, method="mean/sigma")
    assert abs(res["A"] - A) < 1e-9
    assert abs(res["B"] - B) < 1e-9


def test_the_identity_link_is_recovered_as_A_1_B_0():
    a = [1.0, 0.9, 1.1]
    b = [-0.5, 0.0, 0.5]
    res = linkmm(a, b, a, b)
    assert abs(res["A"] - 1.0) < 1e-12
    assert abs(res["B"]) < 1e-12


def test_transformed_parameters_land_on_the_target_scale():
    a_from = [1.0, 0.8, 1.2]
    b_from = [-1.0, 0.0, 1.0]
    A, B = 2.0, 0.5
    a_to = [a / A for a in a_from]
    b_to = [A * b + B for b in b_from]
    res = linkmm(a_from, b_from, a_to, b_to)
    for i in range(3):
        assert abs(res["b_transformed"][i] - b_to[i]) < 1e-9
        assert abs(res["a_transformed"][i] - a_to[i]) < 1e-9
    assert res["n_common"] == 3


def test_validation():
    for call in (lambda: linkmm([1.0], [0.0], [1.0], [0.0]),
                 lambda: linkmm([1.0, 1.0], [0.0, 1.0], [1.0], [0.0]),
                 lambda: linkmm([1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
                                [0.0, 1.0]),
                 lambda: linkmm([1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                                [0.0, 1.0], method="mean/sigma"),
                 lambda: linkmm([1.0, 1.0], [0.0, 1.0], [1.0, 1.0],
                                [0.0, 1.0], method="haebara")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
