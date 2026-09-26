"""nmrcf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmrcf import roll_call_filter


def test_nmrcf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_filter(data=None)
