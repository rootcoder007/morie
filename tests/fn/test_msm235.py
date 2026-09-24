"""Verification tests for msm235.

The stub generator stamped several extracted page fragments with the
same function name, so ksvmdual lives once in msm234 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm234 tests.
"""

import morie.fn.msm234 as host
import morie.fn.msm235 as alias
from morie.fn.msm235 import ksvmdual


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert ksvmdual is getattr(host, "ksvmdual")
    assert alias.ksvmdual is getattr(host, "ksvmdual")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "ksvmdual" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.ksvmdual.__module__ == getattr(host, "ksvmdual").__module__
    assert alias.ksvmdual.__doc__ == getattr(host, "ksvmdual").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm235" in alias.cheatsheet()
