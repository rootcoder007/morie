"""Verification tests for taji_d.

The stub generator stamped several extracted page fragments with the
same function name, so tajimas_d lives once in tajd and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
tajd tests.
"""

import morie.fn.tajd as host
import morie.fn.taji_d as alias
from morie.fn.taji_d import tajimas_d


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert tajimas_d is getattr(host, "tajimas_d")
    assert alias.tajimas_d is getattr(host, "tajimas_d")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "tajimas_d" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.tajimas_d.__module__ == getattr(host, "tajimas_d").__module__
    assert alias.tajimas_d.__doc__ == getattr(host, "tajimas_d").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "taji_d" in alias.cheatsheet()
