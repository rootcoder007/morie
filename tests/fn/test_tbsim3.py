"""tbsim3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbsim3 import tbsim3


def test_tbsim3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbsim3()
