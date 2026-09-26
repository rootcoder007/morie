"""srhm3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srhm3 import srhm3


def test_srhm3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srhm3()
