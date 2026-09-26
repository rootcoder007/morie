"""nbleq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbleq import nbleq


def test_nbleq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbleq()
