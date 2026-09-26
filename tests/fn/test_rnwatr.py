"""rnwatr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnwatr import rnwatr


def test_rnwatr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnwatr()
