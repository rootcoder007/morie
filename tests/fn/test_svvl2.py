"""svvl2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svvl2 import valence_2d


def test_svvl2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        valence_2d(data=None)
