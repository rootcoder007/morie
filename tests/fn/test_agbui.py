"""agbui is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agbui import agbui


def test_agbui_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agbui()
