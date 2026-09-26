"""chsim3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chsim3 import chsim3


def test_chsim3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chsim3()
