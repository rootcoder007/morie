"""xrwrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwrs import w_row_std


def test_xrwrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_row_std(data=None)
