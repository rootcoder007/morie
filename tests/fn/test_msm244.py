"""Verification tests for msm244.

The stub generator stamped several extracted page fragments with the
same function name, so mvsml_preprocessing_eq_2_1 lives once in msm240 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm240 tests.
"""

import morie.fn.msm240 as host
import morie.fn.msm244 as alias
from morie.fn.msm244 import mvsml_preprocessing_eq_2_1


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_preprocessing_eq_2_1 is getattr(host, "mvsml_preprocessing_eq_2_1")
    assert alias.mvsml_preprocessing_eq_2_1 is getattr(host, "mvsml_preprocessing_eq_2_1")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "mvsml_preprocessing_eq_2_1" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.mvsml_preprocessing_eq_2_1.__module__ == getattr(host, "mvsml_preprocessing_eq_2_1").__module__
    assert alias.mvsml_preprocessing_eq_2_1.__doc__ == getattr(host, "mvsml_preprocessing_eq_2_1").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm244" in alias.cheatsheet()
