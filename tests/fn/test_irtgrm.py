"""Verification tests for irtgrm.

The stub generator stamped several extracted page fragments with the
same function name, so graded_response_samejima lives once in grmsam and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
grmsam tests.
"""

import morie.fn.grmsam as host
import morie.fn.irtgrm as alias
from morie.fn.irtgrm import graded_response_samejima


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert graded_response_samejima is getattr(host, "graded_response_samejima")
    assert alias.graded_response_samejima is getattr(host, "graded_response_samejima")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "graded_response_samejima" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.graded_response_samejima.__module__ == getattr(host, "graded_response_samejima").__module__
    assert alias.graded_response_samejima.__doc__ == getattr(host, "graded_response_samejima").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "irtgrm" in alias.cheatsheet()
