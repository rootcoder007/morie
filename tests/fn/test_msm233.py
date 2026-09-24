"""Verification tests for msm233.

The stub generator stamped several extracted page fragments with the
same function name, so svmsdual lives once in msm231 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm231 tests.
"""

import morie.fn.msm231 as host
import morie.fn.msm233 as alias
from morie.fn.msm233 import svmsdual


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert svmsdual is getattr(host, "svmsdual")
    assert alias.svmsdual is getattr(host, "svmsdual")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "svmsdual" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.svmsdual.__module__ == getattr(host, "svmsdual").__module__
    assert alias.svmsdual.__doc__ == getattr(host, "svmsdual").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm233" in alias.cheatsheet()
