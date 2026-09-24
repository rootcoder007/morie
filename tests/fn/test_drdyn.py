"""Tests for drdyn.dr_dynamic_did."""

import math

from morie.fn import _array_core as np
from morie.fn.drdyn import dr_dynamic_did


def test_drdyn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_units = 40
    T = 10

    unit_ids = []
    time_ids = []
    cohort_vals = []
    y_vals = []

    for i in range(n_units):
        g = 5.0 if i < 20 else 0.0
        for t in range(1, T + 1):
            unit_ids.append(f"u{i}")
            time_ids.append(float(t))
            cohort_vals.append(g)
            base_val = rng.normal(0, 1)
            if g > 0 and t >= g:
                te = 1.5
            else:
                te = 0.0
            y_vals.append(base_val + te)

    result = dr_dynamic_did(
        y_vals, unit=unit_ids, time=time_ids, cohort=cohort_vals, horizon=3
    )

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "event_time" in result
    assert "att" in result
    assert "n_cells" in result
    assert math.isfinite(result["estimate"])
    assert len(result["event_time"]) == 2 * 3 + 1
    assert len(result["att"]) == 2 * 3 + 1
    assert len(result["n_cells"]) == 2 * 3 + 1


def test_drdyn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_units = 20
    T = 6

    unit_ids = []
    time_ids = []
    cohort_vals = []
    y_vals = []

    for i in range(n_units):
        g = 3.0 if i < 10 else 0.0
        for t in range(1, T + 1):
            unit_ids.append(f"u{i}")
            time_ids.append(float(t))
            cohort_vals.append(g)
            base_val = rng.normal(0, 1)
            if g > 0 and t >= g:
                te = 1.0
            else:
                te = 0.0
            y_vals.append(base_val + te)

    result = dr_dynamic_did(
        y_vals, unit=unit_ids, time=time_ids, cohort=cohort_vals, horizon=1
    )

    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert len(result["event_time"]) == 2 * 1 + 1
    assert len(result["att"]) == 2 * 1 + 1
    assert len(result["n_cells"]) == 2 * 1 + 1
