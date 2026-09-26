"""hyuaa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyuaa import hyuaa


def test_hyuaa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyuaa()
