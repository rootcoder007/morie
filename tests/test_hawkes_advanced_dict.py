"""hawkes_advanced_fit takes a plain dict of columns (round-8 finding)."""

import pytest

from morie.tps_hawkes_advanced import hawkes_advanced_fit


def test_dict_frame_is_accepted():
    r = hawkes_advanced_fit({"OCC_DATE": ["2024-01-01", "2024-01-02"]}, kernel="exponential", baseline="constant")
    # two events are too few to fit; the point is it gets that far instead of AttributeError
    assert any("only 2 timestamps" in w for w in r.warnings)


def test_dict_without_dates_says_so():
    r = hawkes_advanced_fit({"x": [1, 2, 3]})
    assert any("no OCC_DATE or REPORT_DATE column" in w for w in r.warnings)


def test_non_frame_is_refused_in_words():
    with pytest.raises(TypeError, match="data frame or a dict"):
        hawkes_advanced_fit([1, 2, 3])
