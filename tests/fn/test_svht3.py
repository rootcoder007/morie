"""svht3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svht3 import hotelling_3cand


def test_svht3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hotelling_3cand(data=None)
