"""Tests for smplsz.sample_size_calc."""

import math

import pytest

from morie.fn.smplsz import sample_size_calc


def test_smplsz_basic():
    """Cochran: n0 = z^2 S^2 / e^2 = 1.96^2 * 9 / 0.25 = 138.29 with R's
    qnorm(0.975); N = 1000 corrects it to 121.49, i.e. samplingbook's
    S^2 / (e^2 / z^2 + S^2 / N), rounded up to 122."""
    r = sample_size_calc(0.5, 3.0, 1000)
    assert r["z"] == pytest.approx(1.9599639845400536, rel=1e-12)
    assert r["n0"] == pytest.approx(138.29251754498844, rel=1e-12)
    assert r["n0"] / (1 + r["n0"] / 1000) == pytest.approx(121.49119441042336, rel=1e-12)
    assert r["n"] == 122.0


def test_smplsz_edge():
    """An infinite population needs no correction; bad inputs raise."""
    r = sample_size_calc(0.5, 3.0)
    assert r["n"] == math.ceil(r["n0"]) == 139
    assert sample_size_calc(0.5, 3.0, level=0.99)["n"] > r["n"]
    with pytest.raises(ValueError, match="positive"):
        sample_size_calc(0.0, 3.0)
    with pytest.raises(ValueError, match="level"):
        sample_size_calc(0.5, 3.0, level=1.0)


