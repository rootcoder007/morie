"""kgbkd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgbkd import bk_discretize


def test_kgbkd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bk_discretize(data=None)
