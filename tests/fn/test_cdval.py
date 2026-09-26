"""cdval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdval import cdval


def test_cdval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdval()
