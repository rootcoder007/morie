"""rfcat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfcat import rfcat


def test_rfcat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfcat()
