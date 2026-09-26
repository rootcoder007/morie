"""pdoIdx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pdoIdx import pdo


def test_pdoIdx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pdo(sst=None)
