"""nmrcy is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmrcy import roll_call_yea_nay


def test_nmrcy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        roll_call_yea_nay(data=None)
