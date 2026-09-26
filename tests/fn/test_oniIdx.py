"""oniIdx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.oniIdx import oni


def test_oniIdx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oni(sst_n34=None)
