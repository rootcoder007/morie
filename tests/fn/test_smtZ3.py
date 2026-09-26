"""smtZ3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.smtZ3 import smt_solver


def test_smtZ3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smt_solver(formula=None)
