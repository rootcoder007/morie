"""wlhsi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlhsi import wlhsi


def test_wlhsi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlhsi()
