"""abuvr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abuvr import abuvr


def test_abuvr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abuvr()
