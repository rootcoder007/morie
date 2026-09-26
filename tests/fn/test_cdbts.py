"""cdbts is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdbts import cdbts


def test_cdbts_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdbts()
