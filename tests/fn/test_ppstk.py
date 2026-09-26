"""ppstk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppstk import ppstk


def test_ppstk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppstk()
