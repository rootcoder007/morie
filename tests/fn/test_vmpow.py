"""vmpow is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmpow import vmpow


def test_vmpow_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmpow()
