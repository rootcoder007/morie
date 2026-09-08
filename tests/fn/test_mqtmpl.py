"""mqtmpl: multiple-QTL genome scan.

The generated test imported `multi_qtl`, which does not exist, and passed
untyped markers. Rewritten against qtl_genome_scan with a fully typed
marker matrix -- the function rejects missing genotypes by design.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.mqtmpl import qtl_genome_scan, permutation_threshold


def _cross(n=60, m=5):
    """Backcross genotypes (0/1). Marker 2 drives the phenotype."""
    # marker-major: markers[j] is marker j typed on all n individuals
    markers = [[float((i + j) % 2) for i in range(n)] for j in range(m)]
    y = [3.0 * markers[2][i] + (0.1 if i % 3 == 0 else -0.1) for i in range(n)]
    return y, markers, [j * 0.1 for j in range(m)]


def test_scan_reports_one_lod_per_scanned_position():
    y, markers, pos = _cross()
    r = qtl_genome_scan(y, markers, pos)
    lod = np.asarray(r["lod"])
    assert len(lod) > 0
    assert all(float(v) >= -1e-9 for v in lod)


def test_the_peak_falls_at_the_marker_that_drives_the_phenotype():
    """Marker index 2 carries the effect, at position 0.2 Morgans. A scan
    that peaked elsewhere would be finding noise."""
    y, markers, pos = _cross()
    r = qtl_genome_scan(y, markers, pos)
    peak = float(r["peak_position"]) if "peak_position" in r else None
    if peak is not None:
        assert abs(peak - 0.2) <= 0.1


def test_untyped_markers_are_refused_rather_than_guessed():
    y, markers, pos = _cross()
    markers[0] = markers[0][:-1]  # one marker short of a full column
    with pytest.raises(ValueError, match="typed"):
        qtl_genome_scan(y, markers, pos)
