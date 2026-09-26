"""trksp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trksp import trksp


def test_trksp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trksp()
