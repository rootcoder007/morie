"""rfnst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfnst import rfnst


def test_rfnst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfnst()
