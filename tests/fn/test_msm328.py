"""Verification tests for msm328.

The stub generator stamped several extracted page fragments with the
same function name, so mvsml_functional_regression_eq_15_3 lives once in msm327 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm327 tests.
"""

import morie.fn.msm327 as host
import morie.fn.msm328 as alias
from morie.fn.msm328 import mvsml_functional_regression_eq_15_3


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_functional_regression_eq_15_3 is getattr(host, "mvsml_functional_regression_eq_15_3")
    assert alias.mvsml_functional_regression_eq_15_3 is getattr(host, "mvsml_functional_regression_eq_15_3")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "mvsml_functional_regression_eq_15_3" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.mvsml_functional_regression_eq_15_3.__module__ == getattr(host, "mvsml_functional_regression_eq_15_3").__module__
    assert alias.mvsml_functional_regression_eq_15_3.__doc__ == getattr(host, "mvsml_functional_regression_eq_15_3").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm328" in alias.cheatsheet()
