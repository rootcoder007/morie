"""rqtmpl: interval mapping for QTL detection.

The generated test imported `qtl_mapping`, a name that does not exist.
Rewritten against interval_mapping and anchored on the LOD profile.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rqtmpl import interval_mapping, haldane, inverse_haldane


def test_haldane_and_its_inverse_round_trip():
    """r = (1 - e^{-2d})/2, and back. A map function that does not invert
    would corrupt every position it reports."""
    for d in (0.01, 0.1, 0.5, 1.0):
        assert inverse_haldane(haldane(d)) == pytest.approx(d)


def test_lod_profile_covers_the_interval_and_is_non_negative():
    n = 40
    y = [float(i % 5) for i in range(n)]
    left = [i % 2 for i in range(n)]
    right = [(i // 2) % 2 for i in range(n)]
    r = interval_mapping(y, left, right, 0.2, step=0.02)
    pos = [float(p) for p in np.asarray(r["position"])]
    lod = [float(v) for v in np.asarray(r["lod"])]
    assert len(pos) == len(lod)
    assert min(pos) >= 0.0 and max(pos) <= 0.2 + 1e-9
    assert all(v >= -1e-9 for v in lod)


def test_the_reported_peak_is_the_maximum_of_the_profile():
    n = 40
    y = [float(i % 5) for i in range(n)]
    left = [i % 2 for i in range(n)]
    right = [(i // 2) % 2 for i in range(n)]
    r = interval_mapping(y, left, right, 0.2, step=0.02)
    lod = [float(v) for v in np.asarray(r["lod"])]
    assert float(r["peak_lod"]) == pytest.approx(max(lod))
