"""trshr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trshr import trshr


def test_trshr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trshr()
