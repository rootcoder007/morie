"""geeduc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geeduc import geeduc


def test_geeduc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geeduc()
