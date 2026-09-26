"""mtsbd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtsbd import mtsbd


def test_mtsbd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtsbd()
