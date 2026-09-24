"""Verification tests for jrqbst.

The stub generator stamped several extracted page fragments with the
same function name, so jarque_bera lives once in jarber and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
jarber tests.
"""

import morie.fn.jarber as host
import morie.fn.jrqbst as alias
from morie.fn.jrqbst import jarque_bera


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert jarque_bera is getattr(host, "jarque_bera")
    assert alias.jarque_bera is getattr(host, "jarque_bera")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "jarque_bera" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.jarque_bera.__module__ == getattr(host, "jarque_bera").__module__
    assert alias.jarque_bera.__doc__ == getattr(host, "jarque_bera").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "jrqbst" in alias.cheatsheet()
