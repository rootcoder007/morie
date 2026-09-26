"""rsmsr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsmsr import rsmsr


def test_rsmsr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsmsr()
