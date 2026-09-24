"""Verification tests for incidens.

The stub generator stamped several extracted page fragments with the
same function name, so incidence_rate lives once in cdinc and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
cdinc tests.
"""

import morie.fn.cdinc as host
import morie.fn.incidens as alias
from morie.fn.incidens import incidence_rate


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert incidence_rate is getattr(host, "incidence_rate")
    assert alias.incidence_rate is getattr(host, "incidence_rate")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "incidence_rate" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.incidence_rate.__module__ == getattr(host, "incidence_rate").__module__
    assert alias.incidence_rate.__doc__ == getattr(host, "incidence_rate").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "incidens" in alias.cheatsheet()
