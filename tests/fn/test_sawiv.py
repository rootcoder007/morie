"""sawiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawiv import sawiv


def test_sawiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawiv()
