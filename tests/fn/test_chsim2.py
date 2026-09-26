"""chsim2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chsim2 import chsim2


def test_chsim2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chsim2()
