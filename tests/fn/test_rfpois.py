"""rfpois is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfpois import rfpois


def test_rfpois_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfpois()
