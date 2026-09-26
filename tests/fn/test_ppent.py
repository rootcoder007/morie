"""ppent is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppent import ppent


def test_ppent_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppent()
