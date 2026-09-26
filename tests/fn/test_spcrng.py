"""spcrng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcrng import spcrng


def test_spcrng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcrng()
