"""nrst2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nrst2 import nrst2


def test_nrst2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nrst2()
