"""Verification tests for krigsm.

The stub generator stamped several extracted page fragments with the
same function name, so ordinary_kriging lives once in krig and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
krig tests.
"""

import morie.fn.krig as host
import morie.fn.krigsm as alias
from morie.fn.krigsm import ordinary_kriging


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert ordinary_kriging is getattr(host, "ordinary_kriging")
    assert alias.ordinary_kriging is getattr(host, "ordinary_kriging")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "ordinary_kriging" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.ordinary_kriging.__module__ == getattr(host, "ordinary_kriging").__module__
    assert alias.ordinary_kriging.__doc__ == getattr(host, "ordinary_kriging").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "krigsm" in alias.cheatsheet()
