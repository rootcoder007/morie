"""Verification tests for msm185.

The stub generator stamped several extracted page fragments with the
same function name, so wolfedual lives once in msm184 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm184 tests.
"""

import morie.fn.msm184 as host
import morie.fn.msm185 as alias
from morie.fn.msm185 import wolfedual


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert wolfedual is getattr(host, "wolfedual")
    assert alias.wolfedual is getattr(host, "wolfedual")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "wolfedual" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.wolfedual.__module__ == getattr(host, "wolfedual").__module__
    assert alias.wolfedual.__doc__ == getattr(host, "wolfedual").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm185" in alias.cheatsheet()
