"""swasym is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swasym import swasym


def test_swasym_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swasym(W=None)
