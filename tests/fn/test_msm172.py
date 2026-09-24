"""Verification tests for msm172.

The stub generator stamped several extracted page fragments with the
same function name, so hyperpl lives once in msm161 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm161 tests.
"""

import morie.fn.msm161 as host
import morie.fn.msm172 as alias
from morie.fn.msm172 import hyperpl


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert hyperpl is getattr(host, "hyperpl")
    assert alias.hyperpl is getattr(host, "hyperpl")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "hyperpl" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.hyperpl.__module__ == getattr(host, "hyperpl").__module__
    assert alias.hyperpl.__doc__ == getattr(host, "hyperpl").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm172" in alias.cheatsheet()
