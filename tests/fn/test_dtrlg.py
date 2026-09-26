"""dtrlg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtrlg import dtrlg


def test_dtrlg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtrlg()
