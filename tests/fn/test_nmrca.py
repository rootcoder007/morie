"""nmrca is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmrca import roll_call_agree


def test_nmrca_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_agree(data=None)
