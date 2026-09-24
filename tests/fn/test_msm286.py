"""Verification tests for msm286.

The stub generator stamped several extracted page fragments with the
same function name, so pensse lives once in msm277 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm277 tests.
"""

import morie.fn.msm277 as host
import morie.fn.msm286 as alias
from morie.fn.msm286 import pensse


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert pensse is getattr(host, "pensse")
    assert alias.pensse is getattr(host, "pensse")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "pensse" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.pensse.__module__ == getattr(host, "pensse").__module__
    assert alias.pensse.__doc__ == getattr(host, "pensse").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm286" in alias.cheatsheet()
