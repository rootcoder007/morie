"""Tests for polqnt (polar-transformation vector compression).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.polqnt import polarquant


def test_the_radius_is_the_norm():
    x = [3.0, 4.0]
    res = polarquant(x, quantize=False)
    assert abs(res["radius"] - 5.0) < 1e-12
    assert res["n"] == 2


def test_without_quantisation_the_reconstruction_is_exact():
    x = [1.0, -2.0, 0.5, 3.0]
    res = polarquant(x, quantize=False)
    for i in range(4):
        assert abs(res["reconstruction"][i] - x[i]) < 1e-9
    assert res["relative_l2"] < 1e-9


def test_more_bits_reconstruct_more_faithfully():
    x = [1.0, -2.0, 0.5, 3.0, 0.2, -1.1, 2.2, 0.7]
    coarse = polarquant(x, bits_first=2, bits_rest=1)
    fine = polarquant(x, bits_first=8, bits_rest=6)
    assert fine["mse"] < coarse["mse"]
    assert fine["relative_l2"] < coarse["relative_l2"]


def test_the_bit_budget_is_reported():
    x = [1.0, 2.0, 3.0, 4.0]
    res = polarquant(x, bits_first=4, bits_rest=2)
    assert res["bits_per_coord"] > 0
    assert len(res["codes"]) >= 1


def test_a_zero_vector_is_handled():
    res = polarquant([0.0, 0.0], quantize=False)
    assert abs(res["radius"]) < 1e-12
    assert all(abs(v) < 1e-12 for v in res["reconstruction"])


def test_validation():
    for call in (lambda: polarquant([1.0, 2.0, 3.0]),      # not a power of 2
                 lambda: polarquant([1.0]),
                 lambda: polarquant([1.0, 2.0], bits_first=0),
                 lambda: polarquant([1.0, 2.0], bits_rest=0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
