"""samoe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.samoe import samoe


def test_samoe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        samoe()
