"""fobufr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fobufr import fobufr


def test_fobufr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fobufr()
