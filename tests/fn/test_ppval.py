"""ppval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppval import ppval


def test_ppval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppval()
