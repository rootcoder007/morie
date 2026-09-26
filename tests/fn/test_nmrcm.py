"""nmrcm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmrcm import roll_call_matrix


def test_nmrcm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_matrix(data=None)
