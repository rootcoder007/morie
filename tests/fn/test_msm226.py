"""Verification tests for msm226.

The stub generator stamped several extracted page fragments with the
same function name, so svmkkt lives once in msm223 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm223 tests.
"""

import morie.fn.msm223 as host
import morie.fn.msm226 as alias
from morie.fn.msm226 import svmkkt


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert svmkkt is getattr(host, "svmkkt")
    assert alias.svmkkt is getattr(host, "svmkkt")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "svmkkt" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.svmkkt.__module__ == getattr(host, "svmkkt").__module__
    assert alias.svmkkt.__doc__ == getattr(host, "svmkkt").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm226" in alias.cheatsheet()
