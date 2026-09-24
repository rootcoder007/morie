"""Verification tests for msm281.

The stub generator stamped several extracted page fragments with the
same function name, so penmat lives once in msm278 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm278 tests.
"""

import morie.fn.msm278 as host
import morie.fn.msm281 as alias
from morie.fn.msm281 import penmat


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert penmat is getattr(host, "penmat")
    assert alias.penmat is getattr(host, "penmat")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "penmat" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.penmat.__module__ == getattr(host, "penmat").__module__
    assert alias.penmat.__doc__ == getattr(host, "penmat").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm281" in alias.cheatsheet()
