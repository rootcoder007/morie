"""ophdb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ophdb import ophdb


def test_ophdb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ophdb()
