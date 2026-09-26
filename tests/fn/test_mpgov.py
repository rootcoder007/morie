"""mpgov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpgov import mpgov


def test_mpgov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpgov()
