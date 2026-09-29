"""Tests for morie.fn.stars_and_bars: values recomputed from first principles."""

from morie.fn.stars_and_bars import stars_and_bars


def test_count_multisets_by_enumeration():
    import itertools

    for n, N in ((3, 4), (5, 2), (0, 3)):
        want = sum(1 for _ in itertools.combinations_with_replacement(range(N), n))
        assert stars_and_bars(n, N)["count"] == want
