"""afheat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afheat import afheat


def test_afheat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afheat()
