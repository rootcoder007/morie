"""swfro is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swfro import swfro


def test_swfro_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swfro(W=None)
