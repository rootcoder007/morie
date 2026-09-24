"""Verification tests for msm191.

The stub generator stamped several extracted page fragments with the
same function name, so qplincon lives once in msm188 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm188 tests.
"""

import morie.fn.msm188 as host
import morie.fn.msm191 as alias
from morie.fn.msm191 import qplincon


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert qplincon is getattr(host, "qplincon")
    assert alias.qplincon is getattr(host, "qplincon")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "qplincon" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.qplincon.__module__ == getattr(host, "qplincon").__module__
    assert alias.qplincon.__doc__ == getattr(host, "qplincon").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm191" in alias.cheatsheet()
