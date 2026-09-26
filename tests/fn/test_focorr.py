"""focorr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.focorr import focorr


def test_focorr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        focorr()
