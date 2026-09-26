"""sawrl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawrl import sawrl


def test_sawrl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawrl()
