"""mpclp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpclp import mpclp


def test_mpclp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpclp()
