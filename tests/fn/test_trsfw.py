"""trsfw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trsfw import trsfw


def test_trsfw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trsfw()
