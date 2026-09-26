"""cdpost is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdpost import cdpost


def test_cdpost_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdpost()
