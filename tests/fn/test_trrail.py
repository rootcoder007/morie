"""trrail is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trrail import trrail


def test_trrail_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trrail()
