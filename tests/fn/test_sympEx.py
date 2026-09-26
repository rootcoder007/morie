"""sympEx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sympEx import sympy_expand


def test_sympEx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sympy_expand(expr=None)
