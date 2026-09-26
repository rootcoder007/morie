"""hyunt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyunt import hyunt


def test_hyunt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyunt()
