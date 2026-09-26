"""hyret is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyret import hyret


def test_hyret_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyret()
