"""mvtdv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtdv import mvtdv


def test_mvtdv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtdv()
