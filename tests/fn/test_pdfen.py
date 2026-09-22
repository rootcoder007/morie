"""Tests for pdf_estimate."""

from morie.fn import _array_core as np
import pytest

from morie.fn.pdfen import pdf_estimate, pdfen


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    bins = 30
    r = pdf_estimate(x, bins=bins)
    # value is documented as the maximum of the density values
    assert r.value > 0
    # density array length must equal number of bins
    assert len(r.extra["density"]) == bins
    # centres must align element-wise with density
    assert len(r.extra["centres"]) == len(r.extra["density"])
    # method default and sample size are recorded
    assert r.extra["method"] == "histogram"
    assert r.extra["n"] == len(x)
    # max of stored density must equal the reported value (independent recompute)
    assert r.value == max(r.extra["density"])


def test_alias():
    assert pdfen is pdf_estimate


def test_too_few():
    with pytest.raises(ValueError):
        pdf_estimate([1])
