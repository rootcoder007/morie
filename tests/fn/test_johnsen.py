"""Verification tests for johnsen.

The stub generator stamped several extracted page fragments with the
same function name, so johansen_test lives once in johcg and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
johcg tests.
"""

import morie.fn.johcg as host
import morie.fn.johnsen as alias
from morie.fn.johnsen import johansen_test


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert johansen_test is getattr(host, "johansen_test")
    assert alias.johansen_test is getattr(host, "johansen_test")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "johansen_test" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.johansen_test.__module__ == getattr(host, "johansen_test").__module__
    assert alias.johansen_test.__doc__ == getattr(host, "johansen_test").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "johnsen" in alias.cheatsheet()
