"""Verification tests for msm136.

The stub generator stamped several extracted page fragments with the
same function name, so mvsml_categorical_count_eq_8_6 lives once in msm135 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm135 tests.
"""

import morie.fn.msm135 as host
import morie.fn.msm136 as alias
from morie.fn.msm136 import mvsml_categorical_count_eq_8_6


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_categorical_count_eq_8_6 is getattr(host, "mvsml_categorical_count_eq_8_6")
    assert alias.mvsml_categorical_count_eq_8_6 is getattr(host, "mvsml_categorical_count_eq_8_6")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "mvsml_categorical_count_eq_8_6" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.mvsml_categorical_count_eq_8_6.__module__ == getattr(host, "mvsml_categorical_count_eq_8_6").__module__
    assert alias.mvsml_categorical_count_eq_8_6.__doc__ == getattr(host, "mvsml_categorical_count_eq_8_6").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm136" in alias.cheatsheet()
