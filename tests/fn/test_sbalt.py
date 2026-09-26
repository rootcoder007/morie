"""sbalt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbalt import sbalt


def test_sbalt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbalt()
