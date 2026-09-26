"""mvtpr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtpr import mvtpr


def test_mvtpr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtpr()
