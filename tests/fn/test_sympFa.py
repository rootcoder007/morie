"""sympFa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sympFa import sympy_factor


def test_sympFa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sympy_factor(expr=None)
