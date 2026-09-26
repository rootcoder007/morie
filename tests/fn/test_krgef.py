"""krgef is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.krgef import krgef


def test_krgef_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        krgef()
