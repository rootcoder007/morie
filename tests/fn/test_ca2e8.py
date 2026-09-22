"""Tests for ca2e8.ca_chapter_2_equation_8."""

import numpy as np

from morie.fn import _array_core as np_a
from morie.fn import _frame_core as pd

from morie.fn.ca2e8 import ca_chapter_2_equation_8


def _expected_b2(r_y1, r_y2, r_12, s_y, s_1, s_2):
    """Independent recomputation of the Weisburd et al. (2022) eq. (2.8) formula.

    b_x2 = ((r_y2 - r_y1 * r_12) / (1 - r_12**2)) * (s_y / s_2)
    """
    return ((r_y2 - r_y1 * r_12) / (1.0 - r_12 ** 2)) * (s_y / s_2)


def test_ca2e8_basic():
    """Test basic functionality with documented six scalar inputs."""
    r_y1, r_y2, r_12 = 0.4, 0.55, 0.3
    s_y, s_1, s_2 = 10.0, 2.0, 4.0

    result = ca_chapter_2_equation_8(r_y1, r_y2, r_12, s_y, s_1, s_2)

    # The headline value lives under the 'b2' key in the payload (RichResult dict).
    assert isinstance(result, dict)
    assert "b2" in result

    # Numeric expectation recomputed from the documented formula.
    expected = _expected_b2(r_y1, r_y2, r_12, s_y, s_1, s_2)
    assert result["b2"] == expected

    # Sanity: a few other documented payload entries.
    assert result["value"] == expected
    assert result["method"] == "Weisburd et al. (2022) eq. (2.8)"


def test_ca2e8_edge():
    """Test edge cases: small / moderate correlation and equal SDs."""
    r_y1, r_y2, r_12 = 0.1, -0.2, 0.0
    s_y, s_1, s_2 = 5.0, 1.0, 1.0

    result = ca_chapter_2_equation_8(r_y1, r_y2, r_12, s_y, s_1, s_2)

    assert isinstance(result, dict)
    assert "b2" in result
    expected = _expected_b2(r_y1, r_y2, r_12, s_y, s_1, s_2)
    assert result["b2"] == expected
