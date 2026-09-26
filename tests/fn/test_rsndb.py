"""rsndb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsndb import rsndb


def test_rsndb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsndb()
