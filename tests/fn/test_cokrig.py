"""Tests for cokrig.cokriging."""

import pytest

from morie.fn.cokrig import cokriging

COORDS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
Z1 = [1.0, 2.0, 3.0, 4.0, 2.5]
Z2 = [0.5, 1.5, 2.5, 3.5, 2.0]


def test_cokrig_basic():
    """Both unbiasedness constraints hold and the predictor interpolates."""
    res = cokriging(COORDS, Z1, Z2, [0.25, 0.75])
    assert len(res["lambda"]) == len(COORDS)
    assert len(res["mu"]) == len(COORDS)
    assert abs(sum(res["lambda"]) - 1.0) < 1e-9     # sum lambda = 1
    assert abs(sum(res["mu"])) < 1e-9               # sum mu     = 0
    assert res["method"] == "ordinary co-kriging"
    assert res["estimate"] == res["prediction"]
    assert 1.0 <= res["prediction"] <= 4.0
    # a nugget-free variogram makes kriging an exact interpolator
    for k, p in enumerate(COORDS):
        at = cokriging(COORDS, Z1, Z2, p)
        assert abs(at["prediction"] - Z1[k]) < 1e-8
        assert abs(at["lambda"][k] - 1.0) < 1e-8
        assert abs(at["variance"]) < 1e-8


def test_cokrig_edge():
    """A zero cross-variogram drops the covariate; bad lengths are rejected."""
    off = cokriging(COORDS, Z1, Z2, [0.25, 0.75], cross_vario=lambda h: 0.0)
    assert all(abs(m) < 1e-9 for m in off["mu"])
    assert abs(sum(off["lambda"]) - 1.0) < 1e-9
    # with the covariate switched off the prediction cannot depend on z2
    other = cokriging(COORDS, Z1, [v * 10.0 + 7.0 for v in Z2],
                      [0.25, 0.75], cross_vario=lambda h: 0.0)
    assert abs(other["prediction"] - off["prediction"]) < 1e-9
    # a constant primary field is reproduced exactly, by sum lambda = 1
    flat = cokriging(COORDS, [5.0] * 5, Z2, [0.3, 0.4],
                     cross_vario=lambda h: 0.0)
    assert abs(flat["prediction"] - 5.0) < 1e-9
    # a supplied model is used: a pure-nugget variogram makes every datum
    # equally informative, so the prediction is the mean of z1
    nug = cokriging(COORDS, Z1, Z2, [0.25, 0.75],
                    model=lambda h: 0.0 if h == 0.0 else 1.0,
                    cross_vario=lambda h: 0.0)
    assert abs(nug["prediction"] - sum(Z1) / len(Z1)) < 1e-9
    assert all(abs(w - 0.2) < 1e-9 for w in nug["lambda"])
    with pytest.raises(ValueError):
        cokriging(COORDS, Z1[:4], Z2, [0.0, 0.0])
    with pytest.raises(ValueError):
        cokriging(COORDS, Z1, Z2[:4], [0.0, 0.0])
