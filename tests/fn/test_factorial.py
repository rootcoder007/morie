"""Tests for morie.fn.factorial: values recomputed from first principles."""

from morie.fn.factorial import factorial


def test_product_definition():
    for n in (0, 1, 7, 20):
        p = 1
        for j in range(2, n + 1):
            p *= j
        assert factorial(n)["factorial"] == p


def test_rejects_negative():
    import pytest

    with pytest.raises(ValueError):
        factorial(-1)
