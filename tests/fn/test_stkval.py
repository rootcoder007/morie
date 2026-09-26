"""stkval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stkval import stkval


def test_stkval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stkval()
