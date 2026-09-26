"""mpfrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpfrg import mpfrg


def test_mpfrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpfrg()
