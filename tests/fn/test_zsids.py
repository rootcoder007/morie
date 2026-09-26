"""zsids is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsids import idw_shepard


def test_zsids_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idw_shepard(data=None)
