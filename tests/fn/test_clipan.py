"""clipan is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clipan import clipan


def test_clipan_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clipan()
