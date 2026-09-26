"""abodr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abodr import abodr


def test_abodr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abodr()
