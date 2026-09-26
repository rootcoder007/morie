"""mpens is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpens import mpens


def test_mpens_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpens()
