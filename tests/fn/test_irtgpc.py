"""Verification tests for irtgpc.

The stub generator stamped several extracted page fragments with the
same function name, so generalized_partial_credit lives once in gpcm and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
gpcm tests.
"""

import morie.fn.gpcm as host
import morie.fn.irtgpc as alias
from morie.fn.irtgpc import generalized_partial_credit


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert generalized_partial_credit is getattr(host, "generalized_partial_credit")
    assert alias.generalized_partial_credit is getattr(host, "generalized_partial_credit")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "generalized_partial_credit" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.generalized_partial_credit.__module__ == getattr(host, "generalized_partial_credit").__module__
    assert alias.generalized_partial_credit.__doc__ == getattr(host, "generalized_partial_credit").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "irtgpc" in alias.cheatsheet()
