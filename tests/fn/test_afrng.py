"""afrng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afrng import afrng


def test_afrng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afrng()
