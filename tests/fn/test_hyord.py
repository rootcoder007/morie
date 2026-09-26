"""hyord is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyord import hyord


def test_hyord_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyord()
