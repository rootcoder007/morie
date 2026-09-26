"""ppbay is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppbay import ppbay


def test_ppbay_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppbay()
