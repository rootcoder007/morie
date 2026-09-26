"""mvtst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtst import mvtst


def test_mvtst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtst()
