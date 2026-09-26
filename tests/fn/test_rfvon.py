"""rfvon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfvon import rfvon


def test_rfvon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfvon()
