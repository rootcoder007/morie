"""Verification tests for msm291.

The stub generator stamped several extracted page fragments with the
same function name, so penfreg lives once in msm283 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm283 tests.
"""

import morie.fn.msm283 as host
import morie.fn.msm291 as alias
from morie.fn.msm291 import penfreg


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert penfreg is getattr(host, "penfreg")
    assert alias.penfreg is getattr(host, "penfreg")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "penfreg" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.penfreg.__module__ == getattr(host, "penfreg").__module__
    assert alias.penfreg.__doc__ == getattr(host, "penfreg").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm291" in alias.cheatsheet()
