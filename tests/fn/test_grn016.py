"""Verification tests for grn016.

The module documents its contract with a worked example carrying the
printed value from the source it cites. These tests execute that
example and require every printed value to reproduce exactly, so the
documented contract is enforced here and not only under
--doctest-modules, which the main suite does not run over tests/fn.
"""

import doctest

import morie.fn.grn016 as module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = doctest.testmod(module, verbose=False, report=False,
                          optionflags=doctest.NORMALIZE_WHITESPACE
                          | doctest.ELLIPSIS)
    assert res.attempted >= 1
    assert res.failed == 0


def test_the_worked_example_exercises_the_public_function():
    names = list(getattr(module, "__all__", []) or [])
    assert names
    docs = [module.__doc__ or ""]
    for name in names:
        docs.append(getattr(module, name).__doc__ or "")
    assert ">>>" in "\n".join(docs)


def test_the_module_carries_its_own_cheatsheet():
    assert "grn016" in module.cheatsheet()
