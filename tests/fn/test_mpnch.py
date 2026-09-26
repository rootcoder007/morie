"""mpnch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpnch import mpnch


def test_mpnch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpnch()
