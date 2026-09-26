"""dktpr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dktpr import dktpr


def test_dktpr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dktpr()
