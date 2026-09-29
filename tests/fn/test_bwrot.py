"""Tests for morie.fn.bwrot — Bandwidth selection via rule-of-thumb."""

import pytest

from morie.fn import _array_core as np
from morie.fn.bwrot import bwrot


@pytest.fixture()
def normal_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


def test_returns_dict(normal_data):
    result = bwrot(normal_data)
    assert isinstance(result, dict)
    for key in ("bandwidth", "sigma", "iqr", "method", "n_obs"):
        assert key in result


def test_bandwidth_positive(normal_data):
    result = bwrot(normal_data)
    assert result["bandwidth"] > 0


def test_silverman_default(normal_data):
    result = bwrot(normal_data)
    assert result["method"] == "silverman"


def test_scott_method(normal_data):
    result = bwrot(normal_data, method="scott")
    assert result["method"] == "scott"
    assert result["bandwidth"] > 0


def test_epanechnikov_wider(normal_data):
    gauss = bwrot(normal_data, kernel="gaussian")
    epan = bwrot(normal_data, kernel="epanechnikov")
    assert epan["bandwidth"] > gauss["bandwidth"]


def test_sigma_positive(normal_data):
    result = bwrot(normal_data)
    assert result["sigma"] > 0


def test_n_obs(normal_data):
    result = bwrot(normal_data)
    assert result["n_obs"] == 200


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 2"):
        bwrot(np.array([1.0]))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        bwrot(np.array([1.0, 2.0]), method="plug-in")


def test_larger_n_smaller_bw():
    rng = np.random.default_rng(7)
    small = bwrot(rng.standard_normal(50))
    large = bwrot(rng.standard_normal(500))
    assert large["bandwidth"] < small["bandwidth"]


def test_silverman_and_scott_rules_recomputed():
    import math

    x = [3.1, -0.4, 2.2, 5.9, 1.7, 0.3, 4.4, 2.8, -1.2, 3.6, 7.5, 1.1]
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    xs = sorted(x)

    def q7(p):
        h = (n - 1) * p
        lo = int(h)
        return xs[lo] + (h - lo) * (xs[lo + 1] - xs[lo])

    iqr = q7(0.75) - q7(0.25)
    assert bwrot(x)["bandwidth"] == pytest.approx(1.06 * min(sd, iqr / 1.34) * n**-0.2, rel=1e-12)
    assert bwrot(x, method="scott")["bandwidth"] == pytest.approx(1.059 * sd * n**-0.2, rel=1e-12)
    d0 = lambda r, mu2: (r / mu2**2) ** 0.2  # noqa: E731
    ratio_unif = d0(0.5, 1 / 3) / d0(1 / (2 * math.sqrt(math.pi)), 1.0)
    assert bwrot(x, kernel="uniform")["bandwidth"] == pytest.approx(bwrot(x)["bandwidth"] * ratio_unif, rel=1e-12)


def test_unknown_kernel_raises():
    with pytest.raises(ValueError, match="Unknown kernel"):
        bwrot(np.array([1.0, 2.0, 3.0]), kernel="cosine")
