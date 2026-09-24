"""Verification tests for nutsmc.

The stub generator stamped several extracted page fragments with the
same function name, so nuts_sampler lives once in bnut and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
bnut tests.
"""

import morie.fn.bnut as host
import morie.fn.nutsmc as alias
from morie.fn.nutsmc import nuts_sampler


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert nuts_sampler is getattr(host, "nuts_sampler")
    assert alias.nuts_sampler is getattr(host, "nuts_sampler")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "nuts_sampler" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.nuts_sampler.__module__ == getattr(host, "nuts_sampler").__module__
    assert alias.nuts_sampler.__doc__ == getattr(host, "nuts_sampler").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "nutsmc" in alias.cheatsheet()
