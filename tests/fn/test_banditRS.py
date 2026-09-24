"""Verification tests for banditRS.

The stub generator stamped several extracted page fragments with the
same function name, so linucb lives once in linucb and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
linucb tests.
"""

import morie
import morie.fn
import morie.fn.banditRS as alias
from morie.fn.banditRS import linucb


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert linucb is morie.fn.linucb


def test_every_shared_name_reaches_the_same_one_function():
    assert "linucb" in alias.__all__
    for name in alias.__all__:
        if hasattr(morie.fn, name):
            assert getattr(alias, name) is getattr(morie.fn, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.linucb.__module__ == morie.fn.linucb.__module__
    assert alias.linucb.__doc__ == morie.fn.linucb.__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "banditRS" in alias.cheatsheet()
