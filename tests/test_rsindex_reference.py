"""rsindex: each index recomputed from its published formula; kappa checked against irr::kappa2 in R."""

import math

import pytest

from morie.fn.rsindex import (
    accuracy_assessment,
    change_vector,
    fractional_vegetation_cover,
    land_surface_temperature,
    ndvi_emissivity,
    red_edge_position,
    shortwave_albedo,
    spectral_index,
    toa_reflectance,
)

B = {
    "blue": 0.04,
    "green": 0.08,
    "red": 0.06,
    "rededge": 0.2,
    "nir": 0.42,
    "swir1": 0.21,
    "swir2": 0.11,
    "r531": 0.051,
    "r570": 0.058,
}


def nd(a, b):
    return (a - b) / (a + b)


WANT = {
    "NDVI": nd(0.42, 0.06),
    "GNDVI": nd(0.42, 0.08),
    "NDWI": nd(0.08, 0.42),
    "MNDWI": nd(0.08, 0.21),
    "NDSI": nd(0.08, 0.21),
    "NDBI": nd(0.21, 0.42),
    "NDMI": nd(0.42, 0.21),
    "NBR": nd(0.42, 0.11),
    "NBR2": nd(0.21, 0.11),
    "NDRE": nd(0.42, 0.2),
    "PRI": nd(0.051, 0.058),
    "EVI": 2.5 * 0.36 / (0.42 + 0.36 - 0.3 + 1.0),
    "SAVI": 1.5 * 0.36 / (0.48 + 0.5),
    "MSAVI": (1.84 - math.sqrt(1.84**2 - 8 * 0.36)) / 2,
    "BAI": 1 / (0.04**2 + 0.36**2),
    "MSR": (7.0 - 1) / math.sqrt(8.0),
    "CIre": 0.42 / 0.2 - 1,
}


@pytest.mark.parametrize("index", sorted(WANT))
def test_index_formulas(index):
    assert spectral_index(index, **B) == pytest.approx(WANT[index], abs=1e-14)


def test_vector_bands_and_docstrings():
    assert spectral_index("SAVI", nir=[0.5, 0.4], red=[0.1, 0.1]) == pytest.approx([0.6 / 1.1, 0.45], abs=1e-15)
    assert spectral_index("NDVI", nir=[0.5, 0.3], red=0.1) == pytest.approx([2 / 3, 0.5], abs=1e-15)
    assert round(spectral_index("NDVI", nir=0.5, red=0.1), 6) == 0.666667
    # MSAVI equals SAVI with the self-adjusting L at nir = red
    assert spectral_index("MSAVI", nir=0.3, red=0.3) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        spectral_index("XYZ", nir=0.1)
    with pytest.raises(ValueError):
        spectral_index("NDVI", nir=[0.1, 0.2], red=[0.1])


def test_conversions():
    assert red_edge_position(0.05, 0.1, 0.35, 0.45) == pytest.approx(700 + 40 * 0.15 / 0.25, abs=1e-12)
    assert toa_reflectance(10000, 2e-5, -0.1, 30.0) == pytest.approx(0.2, abs=1e-14)
    assert toa_reflectance([0, 10000], 2e-5, -0.1, 90.0) == pytest.approx([-0.1, 0.1], abs=1e-14)
    assert fractional_vegetation_cover(0.35) == pytest.approx(0.25, abs=1e-15)
    assert fractional_vegetation_cover([0.1, 0.35, 0.9], squared=False) == pytest.approx([0.0, 0.5, 1.0], abs=1e-15)
    assert ndvi_emissivity([0.1, 0.35, 0.7], [0.2, 0.08, 0.05]) == pytest.approx(
        [0.979 - 0.007, 0.987, 0.99], abs=1e-15
    )
    t = land_surface_temperature(300.0, 0.98)
    assert t == pytest.approx(300.0 / (1 + 10.895 * 300 / 14388 * math.log(0.98)), abs=1e-12)
    assert t > 300.0  # emissivity below one raises the kinetic temperature
    assert shortwave_albedo(0.1, 0.1, 0.3, 0.2, 0.1) == pytest.approx(0.1829, abs=1e-15)


def test_change_vector():
    r = change_vector([[0.1, 0.3], [0.2, 0.2]], [[0.4, 0.7], [0.2, 0.1]])
    assert r.magnitude == pytest.approx([0.5, 0.1], abs=1e-15)
    assert r.direction == pytest.approx([math.degrees(math.atan2(0.4, 0.3)), 270.0], abs=1e-12)
    assert math.isnan(change_vector([[1, 2, 3]], [[1, 2, 4]]).direction[0])


def test_accuracy_assessment():
    ref = [1, 1, 2, 2, 2, 1, 3, 3, 3, 2]
    pred = [1, 2, 2, 2, 1, 1, 3, 2, 3, 2]
    r = accuracy_assessment(ref, pred)
    assert r.confusion == [[2, 1, 0], [1, 3, 1], [0, 0, 2]]
    assert r.overall == pytest.approx(0.7, abs=1e-15)
    pe = (3 * 3 + 5 * 4 + 2 * 3) / 100
    assert r.kappa == pytest.approx((0.7 - pe) / (1 - pe), abs=1e-15)
    # irr::kappa2(cbind(ref, pred))$value
    assert r.kappa == pytest.approx(0.5384615384615385, abs=1e-15)
    assert r.producers == pytest.approx([2 / 3, 3 / 4, 2 / 3], abs=1e-15)
    assert r.users == pytest.approx([2 / 3, 3 / 5, 1.0], abs=1e-15)
