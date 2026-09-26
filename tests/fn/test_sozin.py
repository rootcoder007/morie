"""sozin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sozin import sozin


def test_sozin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sozin()
