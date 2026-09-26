"""dtric is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtric import dtric


def test_dtric_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtric()
