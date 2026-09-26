"""svht2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svht2 import hotelling_2party


def test_svht2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hotelling_2party(data=None)
