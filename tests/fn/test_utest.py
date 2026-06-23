"""Tests for morie.fn.utest -- Mann-Whitney alias."""

from morie.fn.mw import mann_whitney_test
from morie.fn.utest import mann_whitney, utest


class TestUtest:
    def test_alias_imports(self):
        assert callable(utest)

    def test_utest_is_mann_whitney_test(self):
        assert utest is mann_whitney_test

    def test_mann_whitney_alias(self):
        assert mann_whitney is mann_whitney_test
