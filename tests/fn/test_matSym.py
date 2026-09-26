"""matSym is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.matSym import matrix_symbolic


def test_matSym_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        matrix_symbolic(M=None)
