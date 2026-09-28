"""Tests for morie.fn.swqueen."""

from morie.fn.swqueen import swqueen


class TestSwqueen:
    def test_basic(self):
        nrow = 4
        ncol = 4
        result = swqueen(nrow, ncol)
        assert result is not None

    def test_returns_spatial_result(self):
        nrow = 4
        ncol = 4
        result = swqueen(nrow, ncol)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        nrow = 4
        ncol = 4
        result = swqueen(nrow, ncol)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))


def test_swqueen_matches_grid_contiguity():
    from morie.fn.swbuild import grid_contiguity

    r = swqueen(3, 4)
    assert r.extra["W"] == grid_contiguity(3, 4, type="queen")
    # interior cells of a 3 x 4 grid have 8 neighbours, corners 3
    assert r.extra["cardinality"][5] == 8 and r.extra["cardinality"][0] == 3
    assert r.statistic == sum(r.extra["cardinality"])
