"""Tests for morie.fn.classify_events: recompute Morin (2016) from the formula."""

from morie.fn.classify_events import classify_events


def test_classification():
    r = classify_events(0.5, 0.4, 0.2)
    assert r["independent"] is True and r["exclusive"] is False
    r = classify_events(0.3, 0.5, 0.0)
    assert r["independent"] is False and r["exclusive"] is True
    r = classify_events(0.0, 0.5, 0.0)
    assert r["independent"] is True and r["exclusive"] is True
    r = classify_events(0.6, 0.5, 0.25)
    assert r["independent"] is False and r["exclusive"] is False
