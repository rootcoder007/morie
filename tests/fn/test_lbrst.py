"""Verification tests for lbrst.

The stub generator stamped several extracted page fragments with the
same function name, so lilliefors_test lives once in lilf and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
lilf tests.
"""

import morie.fn.lilf as host
import morie.fn.lbrst as alias
from morie.fn.lbrst import lilliefors_test


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert lilliefors_test is getattr(host, "lilliefors_test")
    assert alias.lilliefors_test is getattr(host, "lilliefors_test")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "lilliefors_test" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.lilliefors_test.__module__ == getattr(host, "lilliefors_test").__module__
    assert alias.lilliefors_test.__doc__ == getattr(host, "lilliefors_test").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "lbrst" in alias.cheatsheet()
