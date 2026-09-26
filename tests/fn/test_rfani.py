"""rfani is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfani import rfani


def test_rfani_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfani()
