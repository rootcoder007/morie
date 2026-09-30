import math

import pytest

from morie.fn.research_recording import detection_rate_shift, recording_map

M3 = [[0.6, 0.1, 0.05], [0.25, 0.8, 0.0], [0.1, 0.05, 0.9]]
C3 = [123.0, 456.5, 78.25]


def test_recorded_counts_are_the_matrix_product_and_mass_balances():
    r = recording_map(M3, C3)
    for i in range(3):
        assert r.recorded[i] == pytest.approx(math.fsum(M3[i][j] * C3[j] for j in range(3)), abs=1e-12)
    dropped = [(1 - math.fsum(M3[i][j] for i in range(3))) * C3[j] for j in range(3)]
    assert r.dropped["by_category"] == pytest.approx(dropped, abs=1e-12)
    assert r.recorded_total == pytest.approx(r.true_total - r.dropped["total"], abs=1e-10)
    assert r.regime == "cuffing"
    assert r.ratio_recorded[1] == pytest.approx(r.recorded[1] / r.recorded[0], abs=1e-15)


def test_reclassification_keeps_the_total_and_moves_the_ratio():
    q, ci, cj = 0.3, 100.0, 400.0
    r = recording_map([[1.0, q], [0.0, 1 - q]], [ci, cj])
    assert r.regime == "reclassification"
    assert r.recorded_total == pytest.approx(ci + cj, abs=1e-12)
    assert r.recorded[0] / r.recorded[1] == pytest.approx(ci / cj + q / (1 - q) * (1 + ci / cj), abs=1e-12)
    lab = recording_map([[0.7, 0.0], [0.3, 1.0]], {"robbery": 100, "theft": 400})
    assert lab.labels == ["robbery", "theft"] and lab.recorded_total == 500.0


def test_recording_checks():
    with pytest.raises(ValueError, match="column sums"):
        recording_map([[0.8, 0.5], [0.5, 0.5]], [1, 1])
    with pytest.raises(ValueError, match="square"):
        recording_map([[1.0, 0.0]], [1, 1])


def test_detection_rate_rise():
    s = detection_rate_shift(detected=1234, total=9876, n=1500, d=0.27, q=0.35)
    assert s.rise == pytest.approx(0.35 * 1500 * (1 - 0.27) / 9876, abs=1e-15)
    assert s.rate_after - s.rate_before == pytest.approx(s.rise, abs=1e-12)
    with pytest.raises(ValueError, match="d must lie"):
        detection_rate_shift(10, 100, 50, 1, 0.5)
