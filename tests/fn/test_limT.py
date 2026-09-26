"""limT is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.limT import symbolic_limit


def test_limT_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        symbolic_limit(expr=None, x=None, x0=None)
