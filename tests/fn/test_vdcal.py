"""Tests for vdcal.volume_of_distribution."""

import pytest

from morie.fn.vdcal import fut_from_vss, volume_of_distribution


def test_vdcal_basic():
    """Oie-Tozer with the human constants Vp 0.0436, Ve 0.151, Vr 0.380 L/kg
    and R_E/I 1.4: fu = fut = 0.5 gives 0.0436 * 2.4 + 0.5 (0.151 - 0.0436
    * 1.4) + 0.380 = 0.529620 L/kg, 37.07 L at 70 kg."""
    r = volume_of_distribution(None, 0.5, fut=0.5)
    want = 0.0436 * 2.4 + 0.5 * (0.151 - 0.0436 * 1.4) + 0.380
    assert r["vss"] == pytest.approx(want, rel=1e-14)
    assert round(r["vss"], 6) == 0.52962
    assert r["vss_litres"] == pytest.approx(70 * want, rel=1e-14)


def test_vdcal_edge():
    """Solving backwards returns the tissue binding that produced the
    volume; volumes below the plasma-and-water floor are rejected."""
    v = volume_of_distribution(None, 0.2, fut=0.05)["vss"]
    r = volume_of_distribution(None, 0.2, vss=v, direction="fut")
    assert r["vss"] == pytest.approx(v, rel=1e-14)
    assert fut_from_vss(v, 0.2) == pytest.approx(0.05, rel=1e-14)
    with pytest.raises(ValueError, match="below"):
        fut_from_vss(0.05, 0.2)
    with pytest.raises(ValueError, match="free fraction"):
        volume_of_distribution(None, 1.5, fut=0.5)


