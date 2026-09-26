"""hyshr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyshr import hyshr


def test_hyshr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyshr()
