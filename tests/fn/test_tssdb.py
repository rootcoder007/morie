"""tssdb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssdb import tssdb


def test_tssdb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssdb()
