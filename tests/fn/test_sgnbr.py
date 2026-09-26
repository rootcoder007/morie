"""sgnbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgnbr import sgnbr


def test_sgnbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgnbr()
