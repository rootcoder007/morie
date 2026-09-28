"""Tests for morie.fn.swrook."""

from morie.fn.swrook import swrook


class TestSwrook:
    def test_basic(self):
        nrow = 4
        ncol = 4
        result = swrook(nrow, ncol)
        assert result is not None

    def test_returns_spatial_result(self):
        nrow = 4
        ncol = 4
        result = swrook(nrow, ncol)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        nrow = 4
        ncol = 4
        result = swrook(nrow, ncol)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))


def test_swrook_matches_grid_contiguity():
    from morie.fn.swbuild import grid_contiguity

    r = swrook(3, 4)
    assert r.extra["W"] == grid_contiguity(3, 4, type="rook")
    # interior cells of a 3 x 4 grid have 4 neighbours, corners 2
    assert r.extra["cardinality"][5] == 4 and r.extra["cardinality"][0] == 2
    assert r.statistic == sum(r.extra["cardinality"])
