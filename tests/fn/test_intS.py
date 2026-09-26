"""intS is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.intS import symbolic_integrate


def test_intS_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        symbolic_integrate(expr=None, x=None)
