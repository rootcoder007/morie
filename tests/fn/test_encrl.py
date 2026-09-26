"""encrl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.encrl import encrl


def test_encrl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        encrl()
