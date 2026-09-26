"""sogly is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sogly import sogly


def test_sogly_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sogly()
