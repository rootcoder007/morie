"""Tests for mcnem.mcnem."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mcnem import mcnem


def test_mcnem_uses_only_the_discordant_pairs():
    """McNemar's statistic depends on b and c alone. Change the concordant
    cells freely and the statistic must not move -- that is the test's
    defining feature and what separates it from a chi-square on the table."""
    a = mcnem([[10, 20], [30, 40]])
    b = mcnem([[999, 20], [30, 5]])
    assert float(a["statistic"]) == pytest.approx(float(b["statistic"]), rel=1e-12)


def test_mcnem_matches_the_hand_computed_corrected_statistic():
    """(|b - c| - 1)^2 / (b + c) with b = 20, c = 30 gives 81/50 = 1.62."""
    r = mcnem([[10, 20], [30, 40]], continuity=True)
    assert float(r["statistic"]) == pytest.approx((abs(20 - 30) - 1) ** 2 / 50, rel=1e-9)
    assert float(r["statistic"]) == pytest.approx(1.62, rel=1e-9)
    # Without the correction it is (b - c)^2 / (b + c) = 100/50 = 2.
    assert float(mcnem([[10, 20], [30, 40]], continuity=False)["statistic"]) == pytest.approx(2.0, rel=1e-9)


def test_mcnem_symmetric_discordance_gives_no_evidence():
    r = mcnem([[5, 25], [25, 5]], continuity=False)
    assert float(r["statistic"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["pvalue"]) == pytest.approx(1.0, abs=1e-12)


def test_mcnem_strong_asymmetry_is_significant():
    r = mcnem([[10, 40], [5, 10]])
    assert float(r["pvalue"]) < 0.001


def test_mcnem_rejects_a_table_that_is_not_2x2():
    with pytest.raises(ValueError, match="2x2"):
        mcnem([[1, 2, 3], [4, 5, 6]])
