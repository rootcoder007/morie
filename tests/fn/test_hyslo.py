"""hyslo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyslo import hyslo


def test_hyslo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyslo()
