"""enuvb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enuvb import enuvb


def test_enuvb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enuvb()
