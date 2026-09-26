"""sympRe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sympRe import sympy_simplify


def test_sympRe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sympy_simplify(expr=None)
