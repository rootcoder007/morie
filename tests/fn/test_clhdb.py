"""clhdb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clhdb import clhdb


def test_clhdb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clhdb()
