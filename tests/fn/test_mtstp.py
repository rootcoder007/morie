"""mtstp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtstp import mtstp


def test_mtstp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtstp()
