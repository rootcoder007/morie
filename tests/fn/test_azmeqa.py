"""azmeqa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.azmeqa import azmeqa


def test_azmeqa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        azmeqa()
