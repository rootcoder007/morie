"""sgpost is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgpost import sgpost


def test_sgpost_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgpost()
