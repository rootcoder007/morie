"""Verification tests for kmadap.

Kamath, Keenan, Somers and Sorenson (2024), the Houlsby adapter. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.kmadap import kamath_houlsby_adapter


def _gelu(v):
    """Exact GELU: v * Phi(v), with Phi written through erf."""
    return v * 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))


def test_the_adapter_is_a_residual_around_a_bottleneck():
    # adapter(h) = h + W_up GELU(W_down h), with m << d
    h = [1.0, 2.0, 3.0, 4.0]
    W_down = [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]]
    W_up = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    res = kamath_houlsby_adapter(h, W_down, W_up)
    down = [sum(W_down[i][j] * h[j] for j in range(4)) for i in range(2)]
    act = [_gelu(v) for v in down]
    up = [sum(W_up[i][k] * act[k] for k in range(2)) for i in range(4)]
    expected = [h[i] + up[i] for i in range(4)]
    got = [float(v) for v in res["h_adapted"][0]]
    for a, b in zip(got, expected):
        assert a == pytest.approx(b, rel=1e-10)
    assert res["m"] == 2
    assert res["d"] == 4


def test_the_bottleneck_activation_is_gelu_of_the_projection():
    h = [1.0, 2.0, 3.0, 4.0]
    W_down = [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]]
    W_up = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    res = kamath_houlsby_adapter(h, W_down, W_up)
    assert float(res["bottleneck"][0][0]) == pytest.approx(_gelu(0.5), rel=1e-10)
    assert float(res["bottleneck"][0][1]) == pytest.approx(_gelu(1.0), rel=1e-10)


def test_a_bottleneck_no_smaller_than_the_model_is_refused():
    # a Houlsby adapter exists to be cheap: m must be below d
    h = [1.0, 2.0]
    with pytest.raises(ValueError):
        kamath_houlsby_adapter(h, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
