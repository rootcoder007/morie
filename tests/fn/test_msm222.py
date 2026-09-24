"""Verification tests for msm222.

The stub generator stamped several extracted page fragments with the
same function name, so softsvm lives once in msm218 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm218 tests.
"""

import morie.fn.msm218 as host
import morie.fn.msm222 as alias
from morie.fn.msm222 import softsvm


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert softsvm is getattr(host, "softsvm")
    assert alias.softsvm is getattr(host, "softsvm")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "softsvm" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.softsvm.__module__ == getattr(host, "softsvm").__module__
    assert alias.softsvm.__doc__ == getattr(host, "softsvm").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm222" in alias.cheatsheet()
