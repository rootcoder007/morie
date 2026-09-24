"""Verification tests for kpsstst.

The stub generator stamped several extracted page fragments with the
same function name, so kpss_test lives once in kpss and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
kpss tests.
"""

import morie.fn.kpss as host
import morie.fn.kpsstst as alias
from morie.fn.kpsstst import kpss_test


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert kpss_test is getattr(host, "kpss_test")
    assert alias.kpss_test is getattr(host, "kpss_test")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "kpss_test" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.kpss_test.__module__ == getattr(host, "kpss_test").__module__
    assert alias.kpss_test.__doc__ == getattr(host, "kpss_test").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "kpsstst" in alias.cheatsheet()
