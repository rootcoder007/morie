"""Verification tests for msm113.

The stub generator stamped several extracted page fragments with the
same function name, so mvsml_bayesian_regression_pt2_eq_7_9 lives once in msm112 and this module
re-exports it. Its own contract is that the re-exported name reaches
that single object; the arithmetic is verified against the book in the
msm112 tests.
"""

import morie.fn.msm112 as host
import morie.fn.msm113 as alias
from morie.fn.msm113 import mvsml_bayesian_regression_pt2_eq_7_9


def test_the_re_export_is_the_same_object_as_the_implementation():
    assert mvsml_bayesian_regression_pt2_eq_7_9 is getattr(host, "mvsml_bayesian_regression_pt2_eq_7_9")
    assert alias.mvsml_bayesian_regression_pt2_eq_7_9 is getattr(host, "mvsml_bayesian_regression_pt2_eq_7_9")


def test_every_shared_name_reaches_the_same_one_function():
    # names the module defines itself are its own; the ones the host
    # also defines must not have been copied into a second object
    assert "mvsml_bayesian_regression_pt2_eq_7_9" in alias.__all__
    for name in alias.__all__:
        if hasattr(host, name):
            assert getattr(alias, name) is getattr(host, name)


def test_the_alias_carries_the_hosts_documentation_unchanged():
    assert alias.mvsml_bayesian_regression_pt2_eq_7_9.__module__ == getattr(host, "mvsml_bayesian_regression_pt2_eq_7_9").__module__
    assert alias.mvsml_bayesian_regression_pt2_eq_7_9.__doc__ == getattr(host, "mvsml_bayesian_regression_pt2_eq_7_9").__doc__


def test_the_alias_module_carries_its_own_cheatsheet():
    assert "msm113" in alias.cheatsheet()
