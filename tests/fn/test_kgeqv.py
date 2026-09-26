"""kgeqv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgeqv import kriging_equiv


def test_kgeqv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_equiv(values=None, x=None)
