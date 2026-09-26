"""sostr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sostr import sostr


def test_sostr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sostr()
