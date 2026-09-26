"""alphsh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.alphsh import alphsh


def test_alphsh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        alphsh()
