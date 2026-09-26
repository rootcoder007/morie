"""nnint is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nnint import nnint


def test_nnint_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nnint()
