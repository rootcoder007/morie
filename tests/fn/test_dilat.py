"""dilat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dilat import dilat


def test_dilat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dilat()
