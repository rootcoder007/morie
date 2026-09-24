"""Verification tests for zipmd.

The stub generator stamped several extracted page fragments with the
same function name, so zero_inflated_poisson lives once in zinfl and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
zinfl tests.
"""

import morie.fn.zinfl as host
import morie.fn.zipmd as alias
from morie.fn.zipmd import zero_inflated_poisson


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert zero_inflated_poisson is getattr(host, "zero_inflated_poisson")
    assert alias.zero_inflated_poisson is getattr(host, "zero_inflated_poisson")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "zero_inflated_poisson" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.zero_inflated_poisson.__module__ == getattr(host, "zero_inflated_poisson").__module__
    assert alias.zero_inflated_poisson.__doc__ == getattr(host, "zero_inflated_poisson").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "zipmd" in alias.cheatsheet()
