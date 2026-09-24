"""Verification tests for banditRS.

The stub generator stamped several extracted page fragments with the
same function name, so linucb lives once in linucb and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
linucb tests.
"""

import morie.fn.linucb as host
import morie.fn.banditRS as alias
from morie.fn.banditRS import linucb


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert linucb is getattr(host, "linucb")
    assert alias.linucb is getattr(host, "linucb")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "linucb" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.linucb.__module__ == getattr(host, "linucb").__module__
    assert alias.linucb.__doc__ == getattr(host, "linucb").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "banditRS" in alias.cheatsheet()
