"""sawnr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawnr import sawnr


def test_sawnr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawnr()
