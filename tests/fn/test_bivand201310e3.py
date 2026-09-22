"""Tests for bivand201310e3.bivand2013_chapter_10_equation_3."""

import math

from morie.fn import _array_core as np

from morie.fn.bivand201310e3 import bivand2013_chapter_10_equation_3


def test_bivand201310e3_basic():
    """Test basic functionality with hand-crafted data."""
    # Simple, non-random data so we can compute expectations independently.
    O = [5.0, 1.0, 2.0, 0.0, 3.0]
    E = [10.0, 10.0, 10.0, 10.0, 10.0]
    zones = [[0], [1], [2], [3], [4], [0, 1], [2, 3, 4]]

    result = bivand2013_chapter_10_equation_3(O, E, zones)

    # Must be a dict-like RichResult.
    assert isinstance(result, dict)

    # Documented return keys.
    for key in ("loglr", "best", "maxloglr", "bestzone", "Oz", "Ez",
                "rrin", "rrout", "Otot", "Etot", "nzone"):
        assert key in result

    # Per-zone bookkeeping matches what we supplied.
    assert result["nzone"] == len(zones)
    assert len(result["loglr"]) == len(zones)
    assert len(result["Oz"]) == len(zones)
    assert len(result["Ez"]) == len(zones)

    # Totals match an independent sum.
    assert result["Otot"] == sum(O)
    assert result["Etot"] == sum(E)

    # Per-zone Oz/Ez match independent sums over each zone.
    for zs, oz_expected, ez_expected in zip(
        zones, result["Oz"], result["Ez"]
    ):
        assert oz_expected == sum(O[t] for t in zs)
        assert ez_expected == sum(E[t] for t in zs)

    # Independently compute the log-likelihood ratio per zone using
    # the documented formula and compare (allowing tiny float error).
    rr = result["Otot"] / result["Etot"]
    for zs, ll_actual in zip(zones, result["loglr"]):
        oz = sum(O[t] for t in zs)
        ez = sum(E[t] for t in zs)
        oo = result["Otot"] - oz
        eo = result["Etot"] - ez
        # highonly=True: zones whose internal risk doesn't exceed
        # overall risk are scored -inf.
        if ez <= 0.0 or eo <= 0.0:
            expected = float("-inf")
        elif oz / ez <= rr:
            expected = float("-inf")
        else:
            v = 0.0
            if oz > 0.0:
                v += oz * (math.log(oz) - math.log(ez))
            if oo > 0.0:
                v += oo * (math.log(oo) - math.log(eo))
            expected = v
        if math.isinf(expected) and expected < 0:
            assert math.isinf(ll_actual) and ll_actual < 0
        else:
            assert math.isclose(ll_actual, expected, rel_tol=1e-9, abs_tol=1e-12)

    # maxloglr is the max of the loglr list and best points there.
    assert result["maxloglr"] == max(result["loglr"])
    assert result["best"] == result["loglr"].index(result["maxloglr"])
    assert result["bestzone"] == [int(t) for t in zones[result["best"]]]


def test_bivand201310e3_edge():
    """Test edge cases: highonly=False keeps all valid windows."""
    O = [5.0, 1.0, 2.0, 0.0, 3.0]
    E = [10.0, 10.0, 10.0, 10.0, 10.0]
    zones = [[0], [1], [2], [3], [4]]

    result = bivand2013_chapter_10_equation_3(O, E, zones, highonly=False)
    assert isinstance(result, dict)
    assert result["nzone"] == len(zones)

    # With highonly=False, no zone should score -inf from the
    # one-sided restriction (Ez and Eo are all positive here).
    for ll_val in result["loglr"]:
        assert not (math.isinf(ll_val) and ll_val < 0)

    # maxloglr equals max(loglr).
    assert result["maxloglr"] == max(result["loglr"])
