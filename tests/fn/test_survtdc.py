"""Tests for survtdc.time_dep_concordance."""

import math

import pytest

from morie.fn.survtdc import time_dep_concordance


TIME = [1 + ((i * 37) % 59) / 7 + 0.01 * i for i in range(60)]
EVENT = [0 if (i * 3) % 5 == 0 else 1 for i in range(60)]
MARKER = [-TIME[i] + 2 * math.sin(i) + ((i * 11) % 7) / 3 for i in range(60)]
MARKER[4], MARKER[8] = MARKER[5], MARKER[9]


def test_survtdc_basic():
    """Equals R survival::concordance(Surv(time, event) ~ marker,
    reverse = TRUE, ymax = t): 607 concordant, 131 discordant and 1 tied
    pair at t = 4 give 0.8220568335588634."""
    r = time_dep_concordance(TIME, EVENT, MARKER, 4.0)
    assert (r["concordant"], r["tied"], r["comparable"]) == (607.0, 1.0, 739)
    assert r["estimate"] == pytest.approx(0.82205683355886339, rel=1e-14)
    assert time_dep_concordance(TIME, EVENT, MARKER, 6.0)["estimate"] == \
        pytest.approx(0.82420091324200917, rel=1e-14)


def test_survtdc_edge():
    """With no horizon it is Harrell's C; a horizon before every event
    leaves no comparable pair."""
    r = time_dep_concordance(TIME, EVENT, MARKER, float("inf"))
    assert r["estimate"] == pytest.approx(0.80183150183150187, rel=1e-14)
    with pytest.raises(ValueError, match="no comparable pairs"):
        time_dep_concordance(TIME, EVENT, MARKER, 0.5)


