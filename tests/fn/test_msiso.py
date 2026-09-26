"""msiso is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msiso import isotonic_reg


def test_msiso_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isotonic_reg(data=None)
