"""afwndb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afwndb import afwndb


def test_afwndb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afwndb()
