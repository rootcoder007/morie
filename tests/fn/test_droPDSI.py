"""droPDSI: Palmer Drought Severity Index.

The generated test imported `pdsi` (never exported) and passed a 100-element
array as `awc`, which is a scalar soil water capacity. Rewritten against
palmer_pdsi with inputs that are physically meaningful -- the old one fed
standard-normal draws as precipitation, i.e. rain of -1.4 mm.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.droPDSI import palmer_pdsi


def test_pdsi_returns_a_series_as_long_as_its_input():
    precip = [50.0, 60.0, 40.0, 30.0, 80.0, 70.0] * 4
    pet = [45.0] * 24
    r = palmer_pdsi(precip, pet, awc=100.0)
    assert len(np.asarray(r["pdsi"])) == len(precip)
    assert len(np.asarray(r["z_index"])) == len(precip)
    assert all(np.isfinite(v) for v in np.asarray(r["pdsi"]))


def test_precipitation_equal_to_demand_is_not_a_drought():
    """When rainfall exactly matches PET every month there is no departure,
    so the index sits at zero rather than drifting."""
    n = 24
    r = palmer_pdsi([45.0] * n, [45.0] * n, awc=100.0)
    assert float(np.asarray(r["departure"])[-1]) == pytest.approx(0.0, abs=1e-9)
    assert float(np.asarray(r["pdsi"])[-1]) == pytest.approx(0.0, abs=1e-9)


def test_a_dry_spell_is_more_negative_than_a_wet_one():
    """The sign convention: deficit is negative, surplus positive."""
    n = 24
    dry = palmer_pdsi([10.0] * n, [60.0] * n, awc=100.0)
    wet = palmer_pdsi([90.0] * n, [40.0] * n, awc=100.0)
    assert float(np.asarray(dry["pdsi"])[-1]) < float(np.asarray(wet["pdsi"])[-1])
